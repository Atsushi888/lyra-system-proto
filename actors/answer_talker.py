# actors/answer_talker.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Mapping, MutableMapping
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


# llm_meta を「ビューが読むキー」中心に、潰さないための復元対象
_PRESERVE_LLM_META_KEYS = (
    "system_prompt_used",
    "emotion_override",
    "world_state",
    "scene_emotion",
    "models",
    "judge",
    "composer",
    "emotion",
    "memory_context",
    "memory_update",
    "emotion_long_term",
    "judge_mode",
    "judge_mode_next",
    "reply_length_mode",
    "composer_style_hint",
    "style_hint",
    # 失敗時の診断
    "models_error",
    "judge_error",
    "composer_error",
    "emotion_error",
    "world_error",
)


def _is_meaningful(v: Any) -> bool:
    if v is None:
        return False
    if v == "":
        return False
    if v == {}:
        return False
    if v == []:
        return False
    return True


def _merge_preserving(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """
    dst を壊さず、src の「意味のある値」を dst に戻す。
    ただし dst 側が既に意味のある値を持っている場合は dst を優先。
    """
    for k in _PRESERVE_LLM_META_KEYS:
        if k in src and _is_meaningful(src.get(k)) and not _is_meaningful(dst.get(k)):
            dst[k] = src[k]


class AnswerTalker:
    """
    Lyra の統合パイプライン司令塔。
    ※重要：画面切替で AnswerTalker が再生成されても llm_meta を潰さない。
    """

    def __init__(
        self,
        persona: Any,
        llm_manager: Optional[LLMManager] = None,
        memory_model: str = "gpt4o",
        state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.persona = persona

        # state は Streamlit session_state を使うのが前提。
        # view から明示で渡されたらそれを使うが、参照先は同じ dict を期待。
        if state is None:
            self.state: MutableMapping[str, Any] = st.session_state
        else:
            # Mapping が来ても、Streamlit の session_state と同じオブジェクトを想定
            self.state = state  # type: ignore[assignment]

        # persona_id を正しく固定（"default" 固定はバグ源）
        self.persona_id: str = str(getattr(self.persona, "char_id", "default") or "default")

        # ---- 既存 llm_meta を退避（InitAI が触っても戻せるように）----
        prev_meta = {}
        try:
            raw_prev = self.state.get("llm_meta")
            if isinstance(raw_prev, dict):
                prev_meta = dict(raw_prev)
        except Exception:
            prev_meta = {}

        # ★ 初期化（ただし既存 llm_meta は後で復元する）
        # ensure_all が llm_meta を作り直す/上書きする可能性があるため、
        # prev_meta をあとでマージして「潰れ」を防ぐ。
        InitAI.ensure_all(state=self.state, persona=self.persona)

        llm_meta = self.state.get("llm_meta")
        if not isinstance(llm_meta, dict):
            llm_meta = {}
            self.state["llm_meta"] = llm_meta

        # 既存 llm_meta を復元（ビュー切替で models 等が消えるのを防ぐ）
        _merge_preserving(llm_meta, prev_meta)

        # 最低限の形はここで補完（破壊しない）
        llm_meta.setdefault("reply_length_mode", str(self.state.get("reply_length_mode", "auto") or "auto"))
        llm_meta.setdefault("judge_mode", str(self.state.get("judge_mode", "normal") or "normal"))
        llm_meta.setdefault("judge_mode_next", llm_meta.get("judge_mode", "normal"))

        self.llm_meta: Dict[str, Any] = llm_meta
        self.state["llm_meta"] = self.llm_meta

        # PersonaAI（persona_id を正しく）
        self.persona_ai = PersonaAI(persona_id=self.persona_id)

        # LLMManager（persona_id を正しく）
        self.llm_manager: LLMManager = llm_manager or LLMManager.get_or_create(persona_id=self.persona_id)

        # パイプライン部品
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

    # -------------------------------------------------

    def speak(
        self,
        messages: List[Dict[str, str]],
        user_text: str = "",
        judge_mode: Optional[str] = None,
    ) -> str:
        if not messages:
            return ""

        # 軽量補修（毎ターン）
        InitAI.ensure_minimum(state=self.state, persona=self.persona)

        # judge_mode の決定と保存
        mode_current = str(
            judge_mode
            or self.state.get("judge_mode")
            or self.llm_meta.get("judge_mode")
            or "normal"
        )
        self.llm_meta["judge_mode"] = mode_current
        self.state["judge_mode"] = mode_current
        try:
            self.judge_ai.set_mode(mode_current)
        except Exception:
            # set_mode が無い実装でも落とさない
            pass

        # reply_length_mode
        length_mode = str(
            self.state.get("reply_length_mode")
            or self.llm_meta.get("reply_length_mode")
            or "auto"
        )
        self.llm_meta["reply_length_mode"] = length_mode
        self.state["reply_length_mode"] = length_mode

        # Persona 情報（任意・失敗しても続行）
        try:
            persona_all = self.persona_ai.get_all(reload=True)
            if isinstance(persona_all, dict):
                self.llm_meta["persona"] = persona_all
                style_hint = persona_all.get("style_hint") or self.llm_meta.get("composer_style_hint", "")
                self.llm_meta["style_hint"] = style_hint
        except Exception as e:
            self.llm_meta["persona_error"] = str(e)

        # Memory context（ビュー表示用に保存）
        try:
            mem_ctx = self.memory_ai.build_memory_context(user_query=user_text or "")
            self.llm_meta["memory_context"] = mem_ctx
        except Exception as e:
            self.llm_meta["memory_context"] = ""
            self.llm_meta["memory_error"] = str(e)

        # Scene payload（ビュー表示用に保存）
        try:
            scene_payload = self.scene_ai.build_emotion_override_payload()
            if isinstance(scene_payload, dict):
                self.llm_meta["world_state"] = scene_payload.get("world_state", {}) or {}
                self.llm_meta["scene_emotion"] = scene_payload.get("scene_emotion", {}) or {}
                self.llm_meta["world_error"] = None
        except Exception as e:
            self.llm_meta["world_state"] = self.llm_meta.get("world_state", {}) or {}
            self.llm_meta["scene_emotion"] = self.llm_meta.get("scene_emotion", {}) or {}
            self.llm_meta["world_error"] = str(e)
            if LYRA_DEBUG:
                st.exception(e)

        # Mixer override（ビュー表示用に必ず保存）
        try:
            emotion_override = self.mixer_ai.build_emotion_override()
            if not isinstance(emotion_override, dict):
                emotion_override = {}
        except Exception as e:
            emotion_override = {}
            self.llm_meta["models_error"] = f"MixerAI failed: {e}"
            if LYRA_DEBUG:
                st.exception(e)

        self.llm_meta["emotion_override"] = emotion_override

        # system_prompt_used（Persona 側で組み立てられるなら使う）
        # ここは “表示できるなら表示する” 目的。無いなら空でOK。
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

        system_prompt_used = base_system_prompt
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

        self.llm_meta["system_prompt_used"] = system_prompt_used

        # messages の system を差し替え（Persona に委譲できるなら委譲）
        if hasattr(self.persona, "replace_system_prompt"):
            try:
                messages_for_models = self.persona.replace_system_prompt(
                    messages=messages,
                    new_system_prompt=system_prompt_used,
                )
            except Exception:
                messages_for_models = list(messages)
        else:
            new_messages = list(messages)
            system_index = None
            for idx, m in enumerate(new_messages):
                if m.get("role") == "system":
                    system_index = idx
                    break
            sys_msg = {"role": "system", "content": system_prompt_used}
            if system_index is None:
                new_messages.insert(0, sys_msg)
            else:
                new_messages[system_index] = sys_msg
            messages_for_models = new_messages

        # # ---- ModelsAI2 ----
        # try:
        #     results = self.models_ai.collect(
        #         messages_for_models,
        #         mode_current=mode_current,
        #         emotion_override=emotion_override,
        #         reply_length_mode=length_mode,
        #     )
        #     self.llm_meta["models"] = results or {}
        #     self.llm_meta["models_error"] = None
        # except Exception as e:
        #     # ここが落ちると「emotion_override まで出るのに models が無い」になる
        #     self.llm_meta["models"] = {}
        #     self.llm_meta["models_error"] = str(e)
        #     if LYRA_DEBUG:
        #         st.error("[AnswerTalker] ModelsAI2.collect failed")
        #         st.exception(e)
        #     # 返答は一応返す（Council 側のフリーズ回避）
        #     self.state["llm_meta"] = self.llm_meta
        #     return "……ごめん、いま少し調子が悪いみたい。もう一回だけお願い。"

        # ---- ModelsAI2 ----
        try:
            results = self.models_ai.collect(
                messages_for_models,
                mode_current=mode_current,
                emotion_override=emotion_override,
                reply_length_mode=length_mode,
            )

            # ★ここ追加：空結果を検知して models_error を立てる
            if not results:
                self.llm_meta["models"] = {}
                self.llm_meta["models_error"] = (
                    "ModelsAI2.collect returned empty results (no exception). "
                    "Check API keys / enabled providers / model list / message format."
                )
            else:
                self.llm_meta["models"] = results
                self.llm_meta["models_error"] = None

        except Exception as e:
            self.llm_meta["models"] = {}
            self.llm_meta["models_error"] = str(e)
            ...

        # ---- Judge ----
        try:
            judge_result = self.judge_ai.run(
                self.llm_meta.get("models", {}),
                user_text=user_text or "",
            )
            self.llm_meta["judge"] = judge_result or {}
            self.llm_meta["judge_error"] = None
        except Exception as e:
            self.llm_meta["judge"] = {}
            self.llm_meta["judge_error"] = str(e)
            judge_result = {}
            if LYRA_DEBUG:
                st.error("[AnswerTalker] JudgeAI3.run failed")
                st.exception(e)

        # ---- Composer ----
        try:
            composed = self.composer_ai.compose(self.llm_meta)
            self.llm_meta["composer"] = composed or {}
            self.llm_meta["composer_error"] = None
        except Exception as e:
            self.llm_meta["composer"] = {}
            self.llm_meta["composer_error"] = str(e)
            composed = {}
            if LYRA_DEBUG:
                st.error("[AnswerTalker] ComposerAI.compose failed")
                st.exception(e)

        final_text = (composed.get("text") if isinstance(composed, dict) else None) or ""
        if not final_text:
            # composer が空なら judge の chosen_text を拾う
            if isinstance(judge_result, dict):
                final_text = (judge_result.get("chosen_text") or "").strip()

        # ---- Emotion ----
        try:
            emotion_res: EmotionResult = self.emotion_ai.analyze(
                composer=composed if isinstance(composed, dict) else {},
                memory_context=str(self.llm_meta.get("memory_context", "") or ""),
                user_text=user_text or "",
            )
            emo_model = EmotionModel(result=emotion_res)
            emo_model.sync_relationship_fields()
            self.llm_meta["emotion"] = emotion_res.to_dict()
            self.llm_meta["emotion_error"] = None
        except Exception as e:
            self.llm_meta["emotion_error"] = str(e)
            if LYRA_DEBUG:
                st.error("[AnswerTalker] EmotionAI.analyze failed")
                st.exception(e)

        # 反映（ビューが必ず見えるように）
        self.state["llm_meta"] = self.llm_meta

        return final_text
