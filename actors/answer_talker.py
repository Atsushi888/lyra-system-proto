# actors/answer_talker.py

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from actors.models_ai import ModelsAI
from actors.judge_ai2 import JudgeAI2
from actors.composer_ai import ComposerAI
from actors.memory_ai import MemoryAI
from llm.llm_manager import LLMManager, LLMModelConfig


class AnswerTalker:
    """
    AI回答パイプラインの司令塔クラス。

    - LLMManager : 利用可能な LLM の一覧・状態管理
    - ModelsAI   : 複数モデルから回答収集（gpt4o / gpt51 / hermes など）
    - JudgeAI2   : どのモデルの回答を採用するかを決定
    - ComposerAI : 採用候補をもとに最終的な返答テキストを生成
    - MemoryAI   : 会話ターンから長期記憶を抽出・保存し、次ターンに文脈を付与

    Persona と conversation_log から組み立てた messages を入力として、
    自前で Memory → Models → Judge → Composer → MemoryUpdate を回し、
    最終返答テキストを返す。
    """

    def __init__(
        self,
        persona_id: str = "default",
        llm_manager: LLMManager | None = None,
        memory_model: str = "gpt51",
    ) -> None:
        # -----------------------------
        # LLMManager の初期化
        # -----------------------------
        if llm_manager is None:
            llm_manager = self._build_default_llm_manager()
        self.llm_manager: LLMManager = llm_manager

        # 互換用: model_props スナップショット
        self.model_props: Dict[str, Dict[str, Any]] = self.llm_manager.get_model_props()

        # -----------------------------
        # llm_meta の初期化（session_state 経由で共有）
        # -----------------------------
        llm_meta = st.session_state.get("llm_meta")
        if not isinstance(llm_meta, dict):
            llm_meta = {}

        llm_meta.setdefault("models", {})
        llm_meta.setdefault("judge", {})
        llm_meta.setdefault("composer", {})
        llm_meta.setdefault("memory_context", "")
        llm_meta.setdefault("memory_update", {})
        llm_meta.setdefault("memory_update_error", "")

        self.llm_meta: Dict[str, Any] = llm_meta
        st.session_state["llm_meta"] = self.llm_meta

        # -----------------------------
        # サブモジュール構築
        # -----------------------------
        self.models_ai = ModelsAI(self.llm_manager)
        self.judge_ai = JudgeAI2(self.llm_manager)
        self.composer_ai = ComposerAI()

        # MemoryAI 初期化（ModelsAI と同じ LLMRouter を共有）
        self.memory_ai = MemoryAI(
            router=self.models_ai.router,
            persona_id=persona_id,
            model_name=memory_model,
        )

    # ============================
    # LLMManager デフォルト定義
    # ============================
    def _build_default_llm_manager(self) -> LLMManager:
        """
        既定の LLM セットを登録した LLMManager を生成する。
        必要な環境変数が足りないモデルは available=False となり、自動で無効扱いになる。
        """
        mgr = LLMManager()

        mgr.register_model(
            LLMModelConfig(
                name="gpt4o",
                router_fn="call_gpt4o",
                label="GPT-4o",
                priority=3.0,
                vendor="openai",
                required_env=["OPENAI_API_KEY"],
            )
        )
        mgr.register_model(
            LLMModelConfig(
                name="gpt51",
                router_fn="call_gpt51",
                label="GPT-5.1",
                priority=2.0,
                vendor="openai",
                required_env=["OPENAI_API_KEY"],
            )
        )
        mgr.register_model(
            LLMModelConfig(
                name="hermes",
                router_fn="call_hermes",
                label="Hermes 4",
                priority=1.0,
                vendor="openrouter",
                required_env=["OPENROUTER_API_KEY"],
            )
        )

        # 将来 Grok 4.1 を追加する場合はここで:
        # mgr.register_model(
        #     LLMModelConfig(
        #         name="grok41",
        #         router_fn="call_grok41",
        #         label="Grok 4.1",
        #         priority=2.0,
        #         vendor="xai",
        #         required_env=["XAI_API_KEY"],
        #     )
        # )

        return mgr

    # ============================
    # 内部ヘルパ
    # ============================
    @staticmethod
    def _extract_last_user_content(messages: List[Dict[str, Any]]) -> str:
        if not isinstance(messages, list):
            return ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
        return ""

    # ============================
    # MemoryAI: コンテキスト付与
    # ============================
    def attach_memory(
        self,
        messages: List[Dict[str, str]],
        user_text: str = "",
    ) -> List[Dict[str, str]]:
        """
        MemoryAI から「このターンで使うべき記憶コンテキスト」を取得し、
        system メッセージとして差し込んだ新しい messages を返す。

        - llm_meta["memory_context"] を更新
        """
        if not self.memory_ai:
            self.llm_meta["memory_context"] = ""
            st.session_state["llm_meta"] = self.llm_meta
            return messages

        query = user_text or self._extract_last_user_content(messages)

        try:
            mem_context = self.memory_ai.build_memory_context(
                user_query=query,
                max_items=5,
            )
            self.llm_meta["memory_context"] = mem_context
            self.llm_meta["memory_update_error"] = ""
        except Exception as e:
            mem_context = ""
            self.llm_meta["memory_context"] = ""
            self.llm_meta["memory_update_error"] = f"build_memory_context error: {e}"

        st.session_state["llm_meta"] = self.llm_meta

        if not mem_context.strip():
            return messages

        # 既存の system の直後に差し込む
        new_messages: List[Dict[str, str]] = []
        inserted = False
        for msg in messages:
            new_messages.append(msg)
            if not inserted and msg.get("role") == "system":
                new_messages.append(
                    {
                        "role": "system",
                        "content": mem_context,
                    }
                )
                inserted = True

        if not inserted:
            # system がなければ先頭に挿入
            new_messages.insert(
                0,
                {
                    "role": "system",
                    "content": mem_context,
                },
            )

        return new_messages

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
        round_id: int = 0,
    ) -> str:
        """
        Actor から messages（および任意で user_text）を受け取り、
        - MemoryAI.build_memory_context（→ messages へ差し込み）
        - ModelsAI.collect
        - JudgeAI2.process
        - ComposerAI.compose
        - MemoryAI.update_from_turn
        を順に実行して「最終返答テキスト」を返す。

        戻り値:
            final_text: str
        """
        if not messages:
            return ""

        # 0. 記憶コンテキストを差し込む
        messages_with_mem = self.attach_memory(messages, user_text=user_text)

        # 1. 複数モデルの回答収集
        self.run_models(messages_with_mem)

        # 2. JudgeAI2 による採択
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

        # 3. ComposerAI による仕上げ
        try:
            composed = self.composer_ai.compose(self.llm_meta)
        except Exception as e:
            fallback = ""
            if isinstance(judge_result, dict):
                fallback = judge_result.get("chosen_text") or ""
            composed = {
                "status": "error",
                "error": str(e),
                "text": fallback,
            }

        self.llm_meta["composer"] = composed

        # 4. 最終返答テキスト
        final_text = ""
        if isinstance(composed, dict):
            final_text = composed.get("text") or ""
        if not final_text and isinstance(judge_result, dict):
            final_text = judge_result.get("chosen_text") or ""

        # 5. MemoryAI による記憶更新
        if self.memory_ai:
            try:
                mem_update = self.memory_ai.update_from_turn(
                    messages=messages_with_mem,
                    final_reply=final_text,
                    round_id=round_id,
                )
                self.llm_meta["memory_update"] = mem_update
                self.llm_meta["memory_update_error"] = ""
            except Exception as e:
                self.llm_meta["memory_update_error"] = f"update_from_turn error: {e}"

        st.session_state["llm_meta"] = self.llm_meta

        return final_text
