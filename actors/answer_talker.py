# actors/answer_talker.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from llm.llm_manager import LLMManager
from actors.models_ai import ModelsAI
from actors.judge_ai2 import JudgeAI2
from actors.composer_ai import ComposerAI
from actors.memory_ai import MemoryAI


class AnswerTalker:
    """
    AI回答パイプラインの司令塔クラス。

    - LLMManager : 利用可能な LLM の定義と呼び出し窓口
    - ModelsAI   : 複数モデルから回答収集（gpt4o / gpt51 / hermes など）
    - JudgeAI2   : どのモデルの回答を採用するかを決定
    - ComposerAI : 採用候補をもとに最終的な返答テキストを生成
    - MemoryAI   : 会話から長期記憶を抽出・蓄積

    ※ このクラスは「一次返答（reply_text）」を受け取らない。
       Persona と conversation_log から組み立てた messages を入力として、
       自前で Models → Judge → Composer → Memory を回し、
       最終的な返答テキストを返す。
    """

    def __init__(
        self,
        persona_id: str = "default",
        llm_manager: Optional[LLMManager] = None,
        memory_model: str = "gpt51",
        composer_refiner_model: str = "gpt51",
    ) -> None:
        self.persona_id = persona_id

        # LLMManager が渡されなければデフォルト構成を自前で作る
        self.llm_manager: LLMManager = llm_manager or self._build_default_llm_manager(
            persona_id=persona_id
        )

        # llm_meta の初期化（session_state 経由で共有）
        llm_meta = st.session_state.get("llm_meta")
        if not isinstance(llm_meta, dict):
            llm_meta = {
                "models": {},          # ModelsAI.collect の結果
                "judge": {},           # JudgeAI2.process の結果
                "composer": {},        # ComposerAI.compose の結果
                "memory_context": "",  # MemoryAI.build_memory_context の結果
                "memory_update": {},   # MemoryAI.update_from_turn の結果
            }

        self.llm_meta: Dict[str, Any] = llm_meta
        st.session_state["llm_meta"] = self.llm_meta

        # サブモジュール
        self.models_ai = ModelsAI(self.llm_manager)
        self.judge_ai = JudgeAI2(self.llm_manager)
        self.composer_ai = ComposerAI(refiner_model=composer_refiner_model)
        self.memory_ai = MemoryAI(
            llm_manager=self.llm_manager,
            persona_id=self.persona_id,
            model_name=memory_model,
        )

    # ============================
    # 内部: デフォルト LLMManager 構築
    # ============================
    def _build_default_llm_manager(self, persona_id: str) -> LLMManager:
        """
        何も指定されなかったときの標準 LLM 構成をここで定義する。

        - gpt4o  : priority 3.0
        - gpt51  : priority 2.0
        - hermes : priority 1.0
        """
        mgr = LLMManager()

        # ↓ LLMManager 側のヘルパーメソッドを利用
        mgr.register_gpt4o(priority=3.0, enabled=True)
        mgr.register_gpt51(priority=2.0, enabled=True)
        mgr.register_hermes(priority=1.0, enabled=True)

        return mgr

    # ============================
    # ModelsAI 呼び出し
    # ============================
    def run_models(self, messages: List[Dict[str, str]]) -> None:
        """
        Persona.build_messages() で組み立てた messages を受け取り、
        ModelsAI.collect() を実行して llm_meta["models"] を更新する。
        """
        if not messages:
            return

        results = self.models_ai.collect(messages)
        self.llm_meta["models"] = results
        st.session_state["llm_meta"] = self.llm_meta

    # ============================
    # 公開インターフェース
    # ============================
    def speak(
        self,
        messages: List[Dict[str, str]],
        user_text: str = "",
        round_id: Optional[int] = None,
    ) -> str:
        """
        Actor から messages（および任意で user_text / round_id）を受け取り、
        - MemoryAI.build_memory_context
        - ModelsAI.collect
        - JudgeAI2.process
        - ComposerAI.compose
        - MemoryAI.update_from_turn
        を順に実行して「最終返答テキスト」を返す。

        戻り値:
            final_text: str
        """
        if not messages:
            # messages が空なら何もできないので空文字を返す
            return ""

        # 0) MemoryAI: 現在の長期記憶からコンテキスト文字列を生成（今はデバッグ用）
        try:
            memory_context = self.memory_ai.build_memory_context(
                user_query=user_text
            )
            self.llm_meta["memory_context"] = memory_context
        except Exception as e:
            # 失敗しても会話自体は続ける
            self.llm_meta["memory_context"] = ""
            self.llm_meta["memory_update_error"] = (
                f"build_memory_context failed: {e}"
            )

        # 1) 複数モデルの回答収集
        self.run_models(messages)

        # 2) JudgeAI2 による採択
        try:
            models = self.llm_meta.get("models", {})
            judge_result = self.judge_ai.process(models)
        except Exception as e:
            judge_result = {
                "status": "error",
                "error": str(e),
                "chosen_model": "",
                "chosen_text": "",
            }

        self.llm_meta["judge"] = judge_result

        # 3) ComposerAI による仕上げ
        try:
            composed = self.composer_ai.compose(self.llm_meta)
        except Exception as e:
            # 失敗時は Judge の chosen_text をフォールバックとして使う
            fallback = ""
            if isinstance(judge_result, dict):
                fallback = judge_result.get("chosen_text") or ""
            composed = {
                "status": "error",
                "error": str(e),
                "text": fallback,
            }

        self.llm_meta["composer"] = composed

        # 4) 最終返答テキストの決定
        final_text = ""
        if isinstance(composed, dict):
            final_text = composed.get("text") or ""
        if not final_text and isinstance(judge_result, dict):
            final_text = judge_result.get("chosen_text") or ""

        # 5) MemoryAI による記憶更新
        try:
            round_val = int(round_id) if round_id is not None else 0
            mem_update = self.memory_ai.update_from_turn(
                messages=messages,
                final_reply=final_text,
                round_id=round_val,
            )
            self.llm_meta["memory_update"] = mem_update
        except Exception as e:
            self.llm_meta["memory_update"] = {
                "status": "error",
                "added": 0,
                "total": 0,
                "error": str(e),
            }

        # 6) llm_meta を session_state に反映
        st.session_state["llm_meta"] = self.llm_meta

        return final_text
