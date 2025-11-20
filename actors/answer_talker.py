# actors/answer_talker.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from actors.models_ai import ModelsAI
from actors.judge_ai2 import JudgeAI2
from actors.composer_ai import ComposerAI
from actors.memory_ai import MemoryAI
from llm.llm_manager import LLMManager
from llm.llm_manager_factory import get_llm_manager


class AnswerTalker:
    """
    AI回答パイプラインの司令塔クラス。

    - ModelsAI:   複数モデルから回答収集（gpt4o / gpt51 / hermes など）
    - JudgeAI2:   どのモデルの回答を採用するかを決定
    - ComposerAI: 採用候補をもとに最終的な返答テキストを生成
    - MemoryAI:   1ターンごとの会話から長期記憶を抽出・保存

    仕様:
      - Persona と conversation_log から組み立てた messages を入力として、
        自前で Models → Judge → Composer → Memory を回し、
        最終的な返答テキストを返す。
    """

    def __init__(
        self,
        persona: Any,
        llm_manager: Optional[LLMManager] = None,
        memory_model: str = "gpt51",
    ) -> None:
        self.persona = persona

        persona_id = getattr(self.persona, "char_id", "default")

        # ★ ここで get_llm_manager() を呼ぶ。
        #    LLMManager.get_or_create() につながるので、
        #    UserView / AnswerTalker / Council など全部が同じインスタンスを共有する。
        self.llm_manager: LLMManager = llm_manager or get_llm_manager(persona_id)

        # ↓ 以下はこれまで通り
        self.model_props: Dict[str, Dict[str, Any]] = self.llm_manager.get_model_props()

        llm_meta = st.session_state.get("llm_meta")
        if not isinstance(llm_meta, dict):
            llm_meta = {
                "models": {},
                "judge": {},
                "composer": {},
                "memory_context": "",
                "memory_update": {},
            }

        self.llm_meta: Dict[str, Any] = llm_meta
        st.session_state["llm_meta"] = self.llm_meta

        self.models_ai = ModelsAI(self.llm_manager)
        self.judge_ai = JudgeAI2(self.model_props)
        self.composer_ai = ComposerAI()
        self.memory_ai = MemoryAI(
            llm_manager=self.llm_manager,
            persona_id=persona_id,
            model_name=memory_model,
        )

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
    ) -> str:
        """
        Actor から messages（および任意で user_text）を受け取り、
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

        # 0) 次ターン用の memory_context を構築（このターンの system に差し込む運用などを想定）
        try:
            mem_ctx = self.memory_ai.build_memory_context(user_query=user_text or "")
        except Exception as e:
            mem_ctx = ""
            # デバッグ用に llm_meta にも残しておく
            self.llm_meta["memory_context_error"] = str(e)
        self.llm_meta["memory_context"] = mem_ctx

        # ① 複数モデルの回答収集
        self.run_models(messages)

        # ② JudgeAI2 による採択
        try:
            models = self.llm_meta.get("models", {})
            judge_result = self.judge_ai.process(models)
        except Exception as e:
            judge_result = {
                "status": "error",
                "error": str(e),
                "chosen_model": "",
                "chosen_text": "",
                "candidates": [],
            }

        self.llm_meta["judge"] = judge_result

        # ③ ComposerAI による仕上げ
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
                "source_model": judge_result.get("chosen_model") if isinstance(judge_result, dict) else "",
                "mode": "judge_fallback",
            }

        self.llm_meta["composer"] = composed

        # ④ 最終返答テキストの決定
        final_text = ""
        if isinstance(composed, dict):
            final_text = composed.get("text") or ""
        if not final_text and isinstance(judge_result, dict):
            final_text = judge_result.get("chosen_text") or ""

        # ⑤ MemoryAI に、このターンの会話を渡して長期記憶を更新
        try:
            round_val_raw = st.session_state.get("round_number", 0)
            try:
                round_val = int(round_val_raw)
            except Exception:
                round_val = 0

            mem_update = self.memory_ai.update_from_turn(
                messages=messages,
                final_reply=final_text,
                round_id=round_val,
            )
        except Exception as e:
            mem_update = {
                "status": "error",
                "added": 0,
                "total": 0,
                "reason": "exception",
                "raw_reply": "",
                "records": [],
                "error": str(e),
            }

        self.llm_meta["memory_update"] = mem_update

        # ⑥ 最終状態を session_state に反映
        st.session_state["llm_meta"] = self.llm_meta

        return final_text
