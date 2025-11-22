# actors/answer_talker.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from actors.models_ai import ModelsAI
from actors.judge_ai3 import JudgeAI3
from actors.composer_ai import ComposerAI
from actors.memory_ai import MemoryAI
from llm.llm_manager import LLMManager
from llm.llm_manager_factory import get_llm_manager


class AnswerTalker:
    """
    AI回答パイプラインの司令塔クラス。
    """

    def __init__(
        self,
        persona: Any,
        llm_manager: Optional[LLMManager] = None,
        memory_model: str = "gpt4o",
    ) -> None:
        self.persona = persona
        persona_id = getattr(self.persona, "char_id", "default")

        # LLMManager を全体共有
        self.llm_manager: LLMManager = llm_manager or get_llm_manager(persona_id)

        self.model_props: Dict[str, Dict[str, Any]] = self.llm_manager.get_model_props()

        llm_meta = st.session_state.get("llm_meta")
        if not isinstance(llm_meta, dict):
            llm_meta = {
                "models": {},
                "judge": {},
                "judge_mode": "normal",
                "composer": {},
                "memory_context": "",
                "memory_update": {},
            }

        self.llm_meta: Dict[str, Any] = llm_meta
        st.session_state["llm_meta"] = self.llm_meta

        # Multi-LLM 集計
        self.models_ai = ModelsAI(self.llm_manager)

        # ★ JudgeAI3 を導入（初期モードは session の値）
        initial_mode = self.llm_meta.get("judge_mode", "normal")
        self.judge_ai = JudgeAI3(mode=str(initial_mode))

        # Compose & Memory
        self.composer_ai = ComposerAI()
        self.memory_ai = MemoryAI(
            llm_manager=self.llm_manager,
            persona_id=persona_id,
            model_name=memory_model,
        )

    # ---------------------------------------
    # ModelsAI
    # ---------------------------------------
    def run_models(self, messages: List[Dict[str, str]]) -> None:
        if not messages:
            return
        results = self.models_ai.collect(messages)
        self.llm_meta["models"] = results
        st.session_state["llm_meta"] = self.llm_meta

    # ---------------------------------------
    # Main
    # ---------------------------------------
    def speak(
        self,
        messages: List[Dict[str, str]],
        user_text: str = "",
        judge_mode: Optional[str] = None,
    ) -> str:

        if not messages:
            return ""

        # ★ Judge モード指定があれば反映
        if judge_mode is not None:
            try:
                self.judge_ai.set_mode(judge_mode)
                self.llm_meta["judge_mode"] = judge_mode
            except Exception as e:
                self.llm_meta["judge_mode_error"] = str(e)

        # 0) Memory context 生成
        try:
            mem_ctx = self.memory_ai.build_memory_context(user_query=user_text or "")
        except Exception as e:
            mem_ctx = ""
            self.llm_meta["memory_context_error"] = str(e)

        self.llm_meta["memory_context"] = mem_ctx

        # 1) Multi-LLM から回答収集
        self.run_models(messages)

        # 2) JudgeAI3 による採択
        try:
            models = self.llm_meta.get("models", {})
            judge_result = self.judge_ai.run(models)
        except Exception as e:
            judge_result = {
                "status": "error",
                "error": str(e),
                "chosen_model": "",
                "chosen_text": "",
                "candidates": [],
            }

        self.llm_meta["judge"] = judge_result

        # 3) ComposerAI による仕上げ
        try:
            composed = self.composer_ai.compose(self.llm_meta)
        except Exception as e:
            fallback = judge_result.get("chosen_text", "") if isinstance(judge_result, dict) else ""
            composed = {
                "status": "error",
                "error": str(e),
                "text": fallback,
                "source_model": judge_result.get("chosen_model", ""),
                "mode": "judge_fallback",
            }

        self.llm_meta["composer"] = composed

        # 4) 最終返答テキスト
        final_text = composed.get("text") or judge_result.get("chosen_text") or ""

        # 5) MemoryAI 更新
        try:
            round_val = int(st.session_state.get("round_number", 0))
        except Exception:
            round_val = 0

        try:
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
                "error": str(e),
            }

        self.llm_meta["memory_update"] = mem_update
        st.session_state["llm_meta"] = self.llm_meta

        return final_text
