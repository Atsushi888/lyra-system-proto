from __future__ import annotations

from typing import Any, Dict, List, Optional, Mapping
import os
import traceback

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
    """
    ★ デバッグ強化版
    - どの段階で止まっても llm_meta に痕跡を残す
    """

    def __init__(
        self,
        persona: Any,
        llm_manager: Optional[LLMManager] = None,
        memory_model: str = "gpt4o",
        state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.persona = persona
        self.state = state or st.session_state

        InitAI.ensure_all(state=self.state, persona=self.persona)

        self.llm_manager = llm_manager or LLMManager.get_or_create(
            getattr(persona, "char_id", "default")
        )

        self.llm_meta: Dict[str, Any] = self.state.setdefault("llm_meta", {})
        self.llm_meta.setdefault("models", {})
        self.llm_meta.setdefault("errors", [])

        self.persona_ai = PersonaAI(persona_id=getattr(persona, "char_id", "default"))
        self.models_ai = ModelsAI2(self.llm_manager)
        self.emotion_ai = EmotionAI(self.llm_manager, model_name="gpt51")
        self.scene_ai = SceneAI(state=self.state)
        self.mixer_ai = MixerAI(
            state=self.state,
            emotion_ai=self.emotion_ai,
            scene_ai=self.scene_ai,
        )
        self.judge_ai = JudgeAI3(mode="normal")
        self.composer_ai = ComposerAI(self.llm_manager, refine_model="gpt51")
        self.memory_ai = MemoryAI(
            self.llm_manager,
            persona_id=getattr(persona, "char_id", "default"),
            model_name=memory_model,
        )

    # -----------------------------
    # 内部: messages の system に追記
    # -----------------------------
    @staticmethod
    def _inject_into_system(messages: List[Dict[str, str]], extra_system: str) -> List[Dict[str, str]]:
        if not extra_system:
            return messages

        new_messages = list(messages)
        for i, m in enumerate(new_messages):
            if isinstance(m, dict) and m.get("role") == "system":
                base = str(m.get("content") or "")
                merged = (base.rstrip() + "\n\n" + extra_system.strip()).strip()
                new_messages[i] = {"role": "system", "content": merged}
                return new_messages

        # system が無ければ先頭に追加（保険）
        new_messages.insert(0, {"role": "system", "content": extra_system.strip()})
        return new_messages

    def speak(
        self,
        messages: List[Dict[str, str]],
        user_text: str = "",
        judge_mode: Optional[str] = None,
    ) -> str:

        if LYRA_DEBUG:
            st.write(
                "[DEBUG] system_prompt_used exists:",
                "system_prompt_used" in self.llm_meta,
                type(self.llm_meta.get("system_prompt_used")),
                len(self.llm_meta.get("system_prompt_used") or "")
            )

        if not messages:
            return ""

        try:
            InitAI.ensure_minimum(state=self.state, persona=self.persona)

            # ==========================================================
            # ★ NEW: Memory context を作って system へ注入
            # ==========================================================
            memory_context = ""
            try:
                memory_context = self.memory_ai.build_memory_context(user_query=user_text, max_items=5) or ""
                self.llm_meta["memory_context"] = memory_context
            except Exception as e:
                self.llm_meta["memory_context_error"] = str(e)

            if memory_context:
                messages = self._inject_into_system(messages, memory_context)

            # ==========================================================
            # 既存: emotion_override
            # ==========================================================
            emotion_override = self.mixer_ai.build_emotion_override()
            self.llm_meta["emotion_override"] = emotion_override

            # ==========================================================
            # LLMs 実行
            # ==========================================================
            results = self.models_ai.collect(
                messages,
                mode_current=judge_mode or "normal",
                emotion_override=emotion_override,
                reply_length_mode=self.llm_meta.get("reply_length_mode", "auto"),
            )
            self.llm_meta["models"] = results

            if not results:
                raise RuntimeError("ModelsAI2.collect returned empty dict")

            judge = self.judge_ai.run(results, user_text=user_text)
            self.llm_meta["judge"] = judge

            composed = self.composer_ai.compose(self.llm_meta)
            self.llm_meta["composer"] = composed

            final_text = composed.get("text") or judge.get("chosen_text") or ""

            # ==========================================================
            # ★ NEW: Memory 更新（ターン終了時）
            # round_id は streamlit state でインクリメント管理（無ければ作る）
            # ==========================================================
            try:
                rid = int(self.state.get("round_id") or 0)
            except Exception:
                rid = 0
            rid += 1
            try:
                # Mapping の可能性があるので setdefault で落ちないように
                self.state["round_id"] = rid  # type: ignore[index]
            except Exception:
                pass

            try:
                mem_res = self.memory_ai.update_from_turn(
                    messages=messages,
                    final_reply=final_text,
                    round_id=rid,
                )
                self.llm_meta["memory_update"] = mem_res
            except Exception as e:
                self.llm_meta["memory_update_error"] = str(e)

            # ==========================================================
            # emotion（memory_context を渡す）
            # ==========================================================
            try:
                emotion_res: EmotionResult = self.emotion_ai.analyze(
                    composer=composed,
                    memory_context=memory_context or "",
                    user_text=user_text,
                )
                self.llm_meta["emotion"] = emotion_res.to_dict()
                EmotionModel(result=emotion_res).sync_relationship_fields()
            except Exception as e:
                self.llm_meta["emotion_error"] = str(e)

            return final_text

        except Exception as e:
            err = {
                "error": str(e),
                "traceback": traceback.format_exc(limit=8),
            }
            self.llm_meta.setdefault("errors", []).append(err)

            if LYRA_DEBUG:
                st.error("AnswerTalker.speak fatal error")
                st.exception(e)

            return "……（思考が途切れてしまったみたい）"
