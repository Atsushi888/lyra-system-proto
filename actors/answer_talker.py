from __future__ import annotations

from typing import Any, Dict, List, Optional, MutableMapping
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


def _dwrite(*args: Any) -> None:
    if LYRA_DEBUG:
        st.write(*args)


class AnswerTalker:
    """
    Answer パイプライン司令塔。
    - 必ず st.session_state["llm_meta"] に結果を書き戻す
    - persona_id を default 固定しない（char_id を優先）
    """

    def __init__(
        self,
        persona: Any,
        llm_manager: Optional[LLMManager] = None,
        memory_model: str = "gpt4o",
        state: Optional[MutableMapping[str, Any]] = None,
    ) -> None:
        self.persona = persona

        # state が渡されないなら st.session_state を使う（ビューと共有するため）
        self.state: MutableMapping[str, Any] = state if state is not None else st.session_state

        # 最低限のセッション構造を保証
        InitAI.ensure_all(state=self.state, persona=self.persona)

        # persona_id は固定しない
        self.persona_id = str(getattr(self.persona, "char_id", "") or "default")

        # PersonaAI
        self.persona_ai = PersonaAI(persona_id=self.persona_id)

        # LLMManager
        self.llm_manager = llm_manager or LLMManager.get_or_create(persona_id=self.persona_id)

        # llm_meta（必ず session_state 共有）
        llm_meta = self.state.get("llm_meta")
        if not isinstance(llm_meta, dict):
            llm_meta = {}
        self.llm_meta: Dict[str, Any] = llm_meta
        self.state["llm_meta"] = self.llm_meta  # ★共有を保証

        # 必須キー群（ビューが参照するもの）
        self.llm_meta.setdefault("models", {})
        self.llm_meta.setdefault("judge", {})
        self.llm_meta.setdefault("composer", {})
        self.llm_meta.setdefault("emotion", {})
        self.llm_meta.setdefault("memory_context", "")
        self.llm_meta.setdefault("memory_update", {})
        self.llm_meta.setdefault("emotion_override", {})
        self.llm_meta.setdefault("system_prompt_used", "")
        self.llm_meta.setdefault("reply_length_mode", self.state.get("reply_length_mode", "auto") or "auto")
        self.llm_meta.setdefault("judge_mode", self.state.get("judge_mode", "normal") or "normal")
        self.llm_meta.setdefault("judge_mode_next", self.llm_meta.get("judge_mode", "normal"))
        self.state.setdefault("judge_mode", self.llm_meta["judge_mode"])

        # Components
        self.models_ai = ModelsAI2(llm_manager=self.llm_manager)
        self.emotion_ai = EmotionAI(llm_manager=self.llm_manager, model_name="gpt51")
        self.scene_ai = SceneAI(state=self.state)
        self.mixer_ai = MixerAI(state=self.state, emotion_ai=self.emotion_ai, scene_ai=self.scene_ai)
        self.judge_ai = JudgeAI3(mode=str(self.llm_meta.get("judge_mode", "normal") or "normal"))
        self.composer_ai = ComposerAI(llm_manager=self.llm_manager, refine_model="gpt51")
        self.memory_ai = MemoryAI(
            llm_manager=self.llm_manager,
            persona_id=self.persona_id,
            model_name=memory_model,
        )

        _dwrite("[DEBUG:AnswerTalker.__init__] persona_id=", self.persona_id)

    # -------------------------------------------------

    def _replace_system_prompt(self, messages: List[Dict[str, str]], new_system_prompt: str) -> List[Dict[str, str]]:
        """persona.replace_system_prompt が無い場合のフォールバック。"""
        new_messages = list(messages)
        idx = None
        for i, m in enumerate(new_messages):
            if m.get("role") == "system":
                idx = i
                break
        sys_msg = {"role": "system", "content": new_system_prompt}
        if idx is None:
            new_messages.insert(0, sys_msg)
        else:
            new_messages[idx] = sys_msg
        return new_messages

    def speak(
        self,
        messages: List[Dict[str, str]],
        user_text: str = "",
        judge_mode: Optional[str] = None,
    ) -> str:
        if not messages:
            return ""

        # ターンごとに最低限の形が崩れていないか補修
        InitAI.ensure_minimum(state=self.state, persona=self.persona)

        # 共有 llm_meta を毎回 st.session_state に確実に戻す
        self.state["llm_meta"] = self.llm_meta

        # 0) persona 情報（任意）
        try:
            persona_all = self.persona_ai.get_all(reload=True)
            self.llm_meta["persona"] = persona_all
            style_hint = persona_all.get("style_hint") or self.llm_meta.get("composer_style_hint", "")
            self.llm_meta["style_hint"] = style_hint
        except Exception as e:
            self.llm_meta["persona_error"] = str(e)

        # 1) judge_mode / length_mode
        mode_current = str(
            judge_mode
            or self.state.get("judge_mode")
            or self.llm_meta.get("judge_mode")
            or "normal"
        )
        self.judge_ai.set_mode(mode_current)
        self.llm_meta["judge_mode"] = mode_current
        self.state["judge_mode"] = mode_current

        length_mode = str(self.state.get("reply_length_mode") or self.llm_meta.get("reply_length_mode") or "auto")
        self.llm_meta["reply_length_mode"] = length_mode

        _dwrite("[DEBUG:AnswerTalker.speak] mode_current=", mode_current, "length_mode=", length_mode)

        # 2) Scene payload（view 用に積む）
        try:
            scene_payload = self.scene_ai.build_emotion_override_payload()
            self.llm_meta["world_state"] = scene_payload.get("world_state", {}) or {}
            self.llm_meta["scene_emotion"] = scene_payload.get("scene_emotion", {}) or {}
            self.llm_meta["world_error"] = None
        except Exception as e:
            self.llm_meta["world_error"] = str(e)
            self.llm_meta.setdefault("world_state", {})
            self.llm_meta.setdefault("scene_emotion", {})

        # 3) Mixer override（view 用に積む）
        try:
            emotion_override = self.mixer_ai.build_emotion_override()
        except Exception as e:
            emotion_override = {}
            self.llm_meta["mixer_error"] = str(e)
            if LYRA_DEBUG:
                st.exception(e)

        if not isinstance(emotion_override, dict):
            emotion_override = {}

        self.llm_meta["emotion_override"] = emotion_override

        # 4) system_prompt 構築（感情・長さ反映）
        base_system_prompt = ""
        for m in messages:
            if m.get("role") == "system":
                base_system_prompt = m.get("content", "") or ""
                break
        if not base_system_prompt and hasattr(self.persona, "get_system_prompt"):
            try:
                base_system_prompt = str(self.persona.get_system_prompt())
            except Exception:
                base_system_prompt = ""

        if hasattr(self.persona, "build_emotion_based_system_prompt"):
            try:
                system_prompt_used = self.persona.build_emotion_based_system_prompt(
                    base_system_prompt=base_system_prompt,
                    emotion_override=emotion_override,
                    mode_current=mode_current,
                    length_mode=length_mode,
                )
            except Exception:
                system_prompt_used = base_system_prompt
        else:
            system_prompt_used = base_system_prompt

        self.llm_meta["system_prompt_used"] = system_prompt_used or ""

        # messages の system を差し替え（ModelsAI2 が確実に system を受け取るように）
        if hasattr(self.persona, "replace_system_prompt"):
            try:
                messages_for_models = self.persona.replace_system_prompt(messages=messages, new_system_prompt=system_prompt_used)
            except Exception:
                messages_for_models = self._replace_system_prompt(messages, system_prompt_used)
        else:
            messages_for_models = self._replace_system_prompt(messages, system_prompt_used)

        # 5) ModelsAI2.collect（★ここが空のままにならないように “必ず書く”）
        try:
            results = self.models_ai.collect(
                messages_for_models,
                mode_current=mode_current,
                emotion_override=emotion_override,
                reply_length_mode=length_mode,
            )
        except Exception as e:
            # view で原因が見えるように llm_meta に積む
            self.llm_meta["models_error"] = str(e)
            self.llm_meta["models_traceback"] = traceback.format_exc(limit=8)
            self.llm_meta["models"] = {}
            self.state["llm_meta"] = self.llm_meta
            if LYRA_DEBUG:
                st.exception(e)
            return "……ごめん、少し調子が悪いみたい。"

        # collect が dict で空の場合も異常として扱う（原因追跡用）
        if not isinstance(results, dict) or len(results) == 0:
            self.llm_meta["models_error"] = "ModelsAI2.collect returned empty results"
            self.llm_meta["models"] = {}
            self.state["llm_meta"] = self.llm_meta
            return "……（返答生成に失敗したみたい。ログを確認してね）"

        self.llm_meta["models"] = results

        # 6) Judge → Composer
        try:
            judge_result = self.judge_ai.run(
                results,
                user_text=user_text or "",
                preferred_length_mode=length_mode,
            )
        except Exception as e:
            judge_result = {"status": "error", "error": str(e), "chosen_model": "", "chosen_text": "", "candidates": []}
            self.llm_meta["judge_error"] = str(e)

        self.llm_meta["judge"] = judge_result

        try:
            composed = self.composer_ai.compose(self.llm_meta)
        except Exception as e:
            composed = {
                "status": "error",
                "error": str(e),
                "text": judge_result.get("chosen_text") or "",
                "source_model": judge_result.get("chosen_model") or "",
                "mode": "judge_fallback",
            }
            self.llm_meta["composer_error"] = str(e)

        self.llm_meta["composer"] = composed

        final_text = (composed.get("text") or judge_result.get("chosen_text") or "").strip()

        # 7) Emotion（view 用）
        try:
            emotion_res: EmotionResult = self.emotion_ai.analyze(
                composer=composed,
                memory_context=str(self.llm_meta.get("memory_context") or ""),
                user_text=user_text or "",
            )
            emo_model = EmotionModel(result=emotion_res)
            emo_model.sync_relationship_fields()
            self.llm_meta["emotion"] = emotion_res.to_dict()

            # 次モード
            try:
                next_mode = self.emotion_ai.decide_judge_mode(emotion_res)
            except Exception:
                next_mode = mode_current

            self.llm_meta["judge_mode_next"] = next_mode
            self.state["judge_mode"] = next_mode

        except Exception as e:
            self.llm_meta["emotion_error"] = str(e)
            if LYRA_DEBUG:
                st.exception(e)

        # 8) 保存（★ここが重要：view が見るのは st.session_state["llm_meta"]）
        self.state["llm_meta"] = self.llm_meta

        return final_text
