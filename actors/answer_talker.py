from __future__ import annotations

from typing import Any, Dict, List, Optional, MutableMapping, Mapping
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
    """
    AnswerTalkerView(views/answertalker_view.py) が参照する llm_meta を
    常に st.session_state["llm_meta"] に反映させることを最優先に設計。

    - system_prompt_used
    - emotion_override
    - models / judge / composer / emotion / memory
    を必ず llm_meta に積む。
    """

    def __init__(
        self,
        persona: Any,
        llm_manager: Optional[LLMManager] = None,
        memory_model: str = "gpt4o",
        state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.persona = persona

        # state が渡っても、最終的には st.session_state と同期して見える必要がある
        # （View は st.session_state["llm_meta"] を見るため）
        self.state: MutableMapping[str, Any]
        if state is None:
            self.state = st.session_state
        else:
            # Mapping の可能性があるので MutableMapping として扱える前提で保持
            # （Streamlitの session_state を渡している想定）
            self.state = state  # type: ignore[assignment]

        # 初期化を必ず実行
        InitAI.ensure_all(state=self.state, persona=self.persona)

        persona_id = getattr(self.persona, "char_id", "default")

        # PersonaAI
        self.persona_ai = PersonaAI(persona_id=persona_id)

        # LLMManager
        # ここは "default" 固定にすると persona 切替で衝突するので persona_id を使う
        self.llm_manager: LLMManager = llm_manager or LLMManager.get_or_create(persona_id=persona_id)

        # llm_meta は「state」と「st.session_state」の両方に同一参照で置く
        # これで Council 経由・View 直叩き・どっちでも同じ llm_meta を見る。
        llm_meta = self.state.get("llm_meta")
        if not isinstance(llm_meta, dict):
            llm_meta = {}
        self.state["llm_meta"] = llm_meta
        st.session_state["llm_meta"] = llm_meta  # ★ View 側の参照元を確実に更新

        self.llm_meta: Dict[str, Any] = llm_meta

        # View が期待するキーを最低限全部用意
        self.llm_meta.setdefault("reply_length_mode", "auto")
        self.llm_meta.setdefault("system_prompt_used", "")
        self.llm_meta.setdefault("emotion_override", {})
        self.llm_meta.setdefault("models", {})
        self.llm_meta.setdefault("judge", {})
        self.llm_meta.setdefault("composer", {})
        self.llm_meta.setdefault("emotion", {})
        self.llm_meta.setdefault("emotion_error", None)
        self.llm_meta.setdefault("memory_context", "")
        self.llm_meta.setdefault("memory_update", {})
        self.llm_meta.setdefault("composer_style_hint", "")
        self.llm_meta.setdefault("judge_mode", "normal")
        self.llm_meta.setdefault("judge_mode_next", "normal")
        self.llm_meta.setdefault("world_state", {})
        self.llm_meta.setdefault("scene_emotion", {})

        # components
        self.models_ai = ModelsAI2(llm_manager=self.llm_manager)
        self.emotion_ai = EmotionAI(llm_manager=self.llm_manager, model_name="gpt51")
        self.scene_ai = SceneAI(state=self.state)
        self.mixer_ai = MixerAI(state=self.state, emotion_ai=self.emotion_ai, scene_ai=self.scene_ai)
        self.judge_ai = JudgeAI3(mode="normal")
        self.composer_ai = ComposerAI(llm_manager=self.llm_manager, refine_model="gpt51")
        self.memory_ai = MemoryAI(
            llm_manager=self.llm_manager,
            persona_id=persona_id,
            model_name=memory_model,
        )

        # 起動時点で同期しておく
        self._sync_llm_meta()

    # -------------------------------------------------
    # 内部ユーティリティ
    # -------------------------------------------------
    def _sync_llm_meta(self) -> None:
        """llm_meta の参照を state と st.session_state に確実に反映する。"""
        self.state["llm_meta"] = self.llm_meta
        st.session_state["llm_meta"] = self.llm_meta

    def _extract_base_system_prompt(self, messages: List[Dict[str, str]]) -> str:
        for m in messages:
            if m.get("role") == "system":
                return m.get("content", "") or ""
        if hasattr(self.persona, "get_system_prompt"):
            try:
                return str(self.persona.get_system_prompt())
            except Exception:
                return ""
        return ""

    def _replace_system_prompt(self, messages: List[Dict[str, str]], new_system: str) -> List[Dict[str, str]]:
        # Persona が差替関数を持っていれば最優先
        if hasattr(self.persona, "replace_system_prompt"):
            try:
                return self.persona.replace_system_prompt(messages=messages, new_system_prompt=new_system)
            except Exception:
                pass

        new_msgs = list(messages)
        idx = None
        for i, m in enumerate(new_msgs):
            if m.get("role") == "system":
                idx = i
                break
        sys_msg = {"role": "system", "content": new_system}
        if idx is None:
            new_msgs.insert(0, sys_msg)
        else:
            new_msgs[idx] = sys_msg
        return new_msgs

    # -------------------------------------------------
    # メイン
    # -------------------------------------------------
    def speak(
        self,
        messages: List[Dict[str, str]],
        user_text: str = "",
        judge_mode: Optional[str] = None,
    ) -> str:
        if not messages:
            return ""

        # 毎ターン最低限補修
        InitAI.ensure_minimum(state=self.state, persona=self.persona)

        # reply_length_mode は View でも表示するので先に同期
        self.llm_meta["reply_length_mode"] = str(
            self.state.get("reply_length_mode")
            or self.llm_meta.get("reply_length_mode")
            or "auto"
        )

        # judge_mode 現在値
        mode_current = str(judge_mode or self.state.get("judge_mode") or self.llm_meta.get("judge_mode") or "normal")
        self.llm_meta["judge_mode"] = mode_current
        self.state["judge_mode"] = mode_current
        self.judge_ai.set_mode(mode_current) if hasattr(self.judge_ai, "set_mode") else None

        # ---- SceneAI payload（world_state/scene_emotion）----
        try:
            scene_payload = self.scene_ai.build_emotion_override_payload()
            self.llm_meta["world_state"] = scene_payload.get("world_state", {}) or {}
            self.llm_meta["scene_emotion"] = scene_payload.get("scene_emotion", {}) or {}
            self.llm_meta["world_error"] = None
        except Exception as e:
            self.llm_meta["world_state"] = {}
            self.llm_meta["scene_emotion"] = {}
            self.llm_meta["world_error"] = str(e)
            if LYRA_DEBUG:
                st.exception(e)

        # ---- Mixer override ----
        try:
            emotion_override = self.mixer_ai.build_emotion_override() or {}
        except Exception as e:
            emotion_override = {}
            self.llm_meta["mixer_error"] = str(e)
            if LYRA_DEBUG:
                st.error("[AnswerTalker] MixerAI failed")
                st.exception(e)

        self.llm_meta["emotion_override"] = emotion_override
        self._sync_llm_meta()

        # ---- system_prompt_used を必ず作る（Viewがここを見る）----
        base_system = self._extract_base_system_prompt(messages)
        system_used = base_system

        # Persona 側の「感情反映 prompt」を使えるなら使う
        if hasattr(self.persona, "build_emotion_based_system_prompt"):
            try:
                system_used = self.persona.build_emotion_based_system_prompt(
                    base_system_prompt=base_system,
                    emotion_override=emotion_override,
                    mode_current=mode_current,
                    length_mode=self.llm_meta["reply_length_mode"],
                )
            except Exception:
                system_used = base_system

        self.llm_meta["system_prompt_used"] = system_used or base_system or ""
        messages_for_models = self._replace_system_prompt(messages, self.llm_meta["system_prompt_used"])
        self._sync_llm_meta()

        # ---- ModelsAI ----
        try:
            results = self.models_ai.collect(
                messages_for_models,
                mode_current=mode_current,
                emotion_override=emotion_override,
                reply_length_mode=self.llm_meta.get("reply_length_mode", "auto"),
            )
        except Exception as e:
            if LYRA_DEBUG:
                st.error("[AnswerTalker] ModelsAI failed")
                st.exception(e)
            # View 側に「落ちた」ことが見えるように保存して返す
            self.llm_meta["models"] = {}
            self.llm_meta["judge"] = {"status": "error", "error": str(e)}
            self.llm_meta["composer"] = {"status": "error", "error": str(e), "text": ""}
            self._sync_llm_meta()
            return "……ごめん、少し調子が悪いみたい。"

        self.llm_meta["models"] = results
        self._sync_llm_meta()

        # ---- Judge ----
        try:
            judge_result = self.judge_ai.run(results, user_text=user_text or "")
        except Exception as e:
            judge_result = {
                "status": "error",
                "error": str(e),
                "chosen_model": "",
                "chosen_text": "",
                "candidates": [],
            }

        self.llm_meta["judge"] = judge_result
        self._sync_llm_meta()

        # ---- Composer ----
        try:
            composed = self.composer_ai.compose(self.llm_meta)
        except Exception as e:
            composed = {
                "status": "error",
                "error": str(e),
                "text": (judge_result.get("chosen_text") or ""),
                "source_model": judge_result.get("chosen_model", ""),
                "mode": "judge_fallback",
            }

        self.llm_meta["composer"] = composed
        self._sync_llm_meta()

        final_text = (composed.get("text") or judge_result.get("chosen_text") or "").strip()

        # ---- EmotionAI（解析結果を View に見せる）----
        try:
            emotion_res: EmotionResult = self.emotion_ai.analyze(
                composer=composed,
                memory_context=self.llm_meta.get("memory_context", "") or "",
                user_text=user_text or "",
            )
            emo_model = EmotionModel(result=emotion_res)
            emo_model.sync_relationship_fields()

            self.llm_meta["emotion"] = emotion_res.to_dict()
            self.llm_meta["emotion_error"] = None

            # judge_mode_next を更新（あれば）
            if hasattr(self.emotion_ai, "decide_judge_mode"):
                try:
                    next_mode = str(self.emotion_ai.decide_judge_mode(emotion_res))
                    self.llm_meta["judge_mode_next"] = next_mode
                    self.state["judge_mode"] = next_mode
                except Exception:
                    pass

        except Exception as e:
            self.llm_meta["emotion_error"] = str(e)
            if LYRA_DEBUG:
                st.exception(e)

        self._sync_llm_meta()
        return final_text
