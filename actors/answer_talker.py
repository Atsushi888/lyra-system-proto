# actors/answer_talker.py

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from actors.models_ai import ModelsAI
from actors.judge_ai2 import JudgeAI2
from actors.composer_ai import ComposerAI
from actors.memory_ai import MemoryAI
from llm.llm_manager import LLMManager


class AnswerTalker:
    """
    AI回答パイプラインの司令塔クラス。

    - LLMManager : 利用可能な LLM の一覧・状態管理
    - ModelsAI   : 複数モデルから回答収集
    - JudgeAI2   : 採用モデルの決定
    - ComposerAI : 採用候補をもとに最終返答テキストを生成
    - MemoryAI   : 会話ターンから長期記憶を抽出・保存し、次ターンに文脈を付与
    """

    def __init__(
        self,
        persona_id: str = "default",
        llm_manager: LLMManager | None = None,
        memory_model: str = "gpt51",
    ) -> None:
        self.persona_id = persona_id

        if llm_manager is None:
            llm_manager = self._build_default_llm_manager(persona_id)
        self.llm_manager: LLMManager = llm_manager

        self.model_props: Dict[str, Dict[str, Any]] = self.llm_manager.get_model_props()

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

        # LLMManager を渡す
        self.models_ai = ModelsAI(self.llm_manager)
        self.judge_ai = JudgeAI2(self.llm_manager)
        self.composer_ai = ComposerAI()

        # MemoryAI には LLMManager 内部の router を渡す
        self.memory_ai = MemoryAI(
            router=self.llm_manager.router,
            persona_id=persona_id,
            model_name=memory_model,
        )

    def _build_default_llm_manager(self, persona_id: str | None = None) -> LLMManager:
        mgr = LLMManager()

        mgr.register_model_gpt4o(priority=3.0)
        mgr.register_model_gpt51(priority=2.0)
        mgr.register_model_hermes4(priority=1.0)

        pid = (persona_id or "").lower()

        if pid == "floria_ja":
            cfg_51 = mgr.get("gpt51")
            if cfg_51:
                cfg_51.priority = 4.0

        elif pid == "succubus_senpai":
            mgr.register_model_grok41(priority=3.5)
            cfg_hermes = mgr.get("hermes")
            if cfg_hermes:
                cfg_hermes.priority = 3.0

        elif pid == "elf_childhood_friend":
            mgr.disable("hermes")

        return mgr

    @staticmethod
    def _extract_last_user_content(messages: List[Dict[str, Any]]) -> str:
        if not isinstance(messages, list):
            return ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
        return ""

    def attach_memory(
        self,
        messages: List[Dict[str, str]],
        user_text: str = "",
    ) -> List[Dict[str, str]]:
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
            new_messages.insert(
                0,
                {
                    "role": "system",
                    "content": mem_context,
                },
            )

        return new_messages

    def run_models(self, messages: List[Dict[str, str]]) -> None:
        if not messages:
            return
        results = self.models_ai.collect(messages)
        self.llm_meta["models"] = results
        st.session_state["llm_meta"] = self.llm_meta

    def speak(
        self,
        messages: List[Dict[str, str]],
        user_text: str = "",
        round_id: int = 0,
    ) -> str:
        if not messages:
            return ""

        messages_with_mem = self.attach_memory(messages, user_text=user_text)

        self.run_models(messages_with_mem)

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

        final_text = ""
        if isinstance(composed, dict):
            final_text = composed.get("text") or ""
        if not final_text and isinstance(judge_result, dict):
            final_text = judge_result.get("chosen_text") or ""

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
