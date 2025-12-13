from __future__ import annotations
from typing import Any, Dict, List, Optional, Mapping
import os
import streamlit as st

from actors.models_ai2 import ModelsAI2
from actors.judge_ai3 import JudgeAI3
from actors.composer_ai import ComposerAI
from actors.memory_ai import MemoryAI
from actors.emotion_ai import EmotionAI, EmotionResult
from actors.emotion.emotion_models import EmotionModel
from actors.persona_ai import PersonaAI
from actors.scene_ai import SceneAI
from actors.mixer_ai import MixerAI
from actors.init_ai import InitAI
from llm.llm_manager import LLMManager

LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"


class AnswerTalker:
    def __init__(
        self,
        persona: Any,
        llm_manager: Optional[LLMManager] = None,
        memory_model: str = "gpt4o",
        state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.persona = persona
        self.state = state or st.session_state

        # ★ 必ず初期化
        InitAI.ensure_all(state=self.state, persona=self.persona)

        self.persona_ai = PersonaAI(persona_id=getattr(persona, "char_id", "default"))
        self.llm_manager = llm_manager or LLMManager.get_or_create(persona_id="default")

        self.llm_meta = self.state.setdefault("llm_meta", {})
        self.llm_meta.setdefault("reply_length_mode", "auto")

        self.models_ai = ModelsAI2(llm_manager=self.llm_manager)
        self.emotion_ai = EmotionAI(llm_manager=self.llm_manager, model_name="gpt51")
        self.scene_ai = SceneAI(state=self.state)
        self.mixer_ai = MixerAI(state=self.state, emotion_ai=self.emotion_ai, scene_ai=self.scene_ai)
        self.judge_ai = JudgeAI3(mode="normal")
        self.composer_ai = ComposerAI(llm_manager=self.llm_manager, refine_model="gpt51")
        self.memory_ai = MemoryAI(llm_manager=self.llm_manager, persona_id="default", model_name=memory_model)

    # -------------------------------------------------

    def speak(
        self,
        messages: List[Dict[str, str]],
        user_text: str = "",
        judge_mode: Optional[str] = None,
    ) -> str:
        if not messages:
            return ""

        InitAI.ensure_minimum(state=self.state, persona=self.persona)

        try:
            scene_payload = self.scene_ai.build_emotion_override_payload()
        except Exception as e:
            scene_payload = {"world_state": {}, "scene_emotion": {}}
            if LYRA_DEBUG:
                st.exception(e)

        try:
            emotion_override = self.mixer_ai.build_emotion_override()
        except Exception as e:
            emotion_override = {}
            st.error("[AnswerTalker] MixerAI failed")
            st.exception(e)

        try:
            results = self.models_ai.collect(
                messages,
                mode_current=judge_mode or "normal",
                emotion_override=emotion_override,
                reply_length_mode=self.llm_meta.get("reply_length_mode", "auto"),
            )
        except Exception as e:
            st.error("[AnswerTalker] ModelsAI failed")
            st.exception(e)
            return "……ごめん、少し調子が悪いみたい。"

        self.llm_meta["models"] = results

        judge_result = self.judge_ai.run(results, user_text=user_text)
        composed = self.composer_ai.compose(self.llm_meta)

        final_text = composed.get("text") or judge_result.get("chosen_text") or ""

        try:
            emotion_res: EmotionResult = self.emotion_ai.analyze(
                composer=composed,
                memory_context="",
                user_text=user_text,
            )
            emo_model = EmotionModel(result=emotion_res)
            emo_model.sync_relationship_fields()
        except Exception as e:
            if LYRA_DEBUG:
                st.exception(e)

        return final_text
