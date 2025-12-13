# actors/answer_talker.py
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
from llm.llm_manager import LLMManager, get_or_create

LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


class AnswerTalker:
    """
    Actor から呼ばれる “回答生成パイプライン”。

    ★重要：
    - llm_meta は st.session_state 共有（ビューが読む）
    - ModelsAI2 の結果を必ず llm_meta["models"] に書く（空でも error を残す）
    - persona_id を default 固定しない（ここが原因で collect が空になる事が多い）
    """

    def __init__(
        self,
        persona: Any,
        llm_manager: Optional[LLMManager] = None,
        memory_model: str = "gpt4o",
        state: Optional[MutableMapping[str, Any]] = None,
    ) -> None:
        self.persona = persona

        # state は “必ず” Streamlit の共有 state を使う（ビューが読む）
        self.state: MutableMapping[str, Any] = state if state is not None else st.session_state

        # persona_id は char_id を最優先（default固定禁止）
        self.persona_id: str = str(getattr(self.persona, "char_id", "default") or "default")

        # ★ 必ず初期化（player_name / world_state / llm_meta 等）
        InitAI.ensure_all(state=self.state, persona=self.persona)

        # llm_meta を共有辞書として確保
        llm_meta = self.state.get("llm_meta")
        if not isinstance(llm_meta, dict):
            llm_meta = {}
            self.state["llm_meta"] = llm_meta
        self.llm_meta: Dict[str, Any] = llm_meta

        # 必須キーを確保（ビューが読む）
        self.llm_meta.setdefault("reply_length_mode", self.state.get("reply_length_mode", "auto") or "auto")
        self.llm_meta.setdefault("models", {})
        self.llm_meta.setdefault("judge", {})
        self.llm_meta.setdefault("composer", {})
        self.llm_meta.setdefault("emotion", {})
        self.llm_meta.setdefault("emotion_override", {})
        self.llm_meta.setdefault("system_prompt_used", "")
        self.llm_meta.setdefault("memory_context", "")
        self.llm_meta.setdefault("memory_update", {})
        self.llm_meta.setdefault("judge_mode", self.state.get("judge_mode", "normal") or "normal")
        self.llm_meta.setdefault("judge_mode_next", self.llm_meta.get("judge_mode", "normal"))

        # PersonaAI（JSON人格）
        self.persona_ai = PersonaAI(persona_id=self.persona_id)

        # ★ LLMManager：persona_id を一致させる（default固定しない）
        self.llm_manager: LLMManager = llm_manager or LLMManager.get_or_create(persona_id=self.persona_id)

        # 各AI
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

        # stateへ反映
        self.state["llm_meta"] = self.llm_meta

    # -------------------------------------------------

    def _replace_system_prompt(self, messages: List[Dict[str, str]], system_prompt: str) -> List[Dict[str, str]]:
        new_messages = list(messages)
        idx = None
        for i, m in enumerate(new_messages):
            if m.get("role") == "system":
                idx = i
                break
        if idx is None:
            new_messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            new_messages[idx] = {"role": "system", "content": system_prompt}
        return new_messages

    def _build_system_prompt_used(
        self,
        messages: List[Dict[str, str]],
        emotion_override: Dict[str, Any],
        mode_current: str,
        length_mode: str,
    ) -> str:
        # base system prompt
        base_system = ""
        for m in messages:
            if m.get("role") == "system":
                base_system = m.get("content", "") or ""
                break
        if not base_system and hasattr(self.persona, "get_system_prompt"):
            try:
                base_system = str(self.persona.get_system_prompt())
            except Exception:
                base_system = ""

        # persona に委譲できるなら委譲
        if hasattr(self.persona, "build_emotion_based_system_prompt"):
            try:
                return str(
                    self.persona.build_emotion_based_system_prompt(
                        base_system_prompt=base_system,
                        emotion_override=emotion_override,
                        mode_current=mode_current,
                        length_mode=length_mode,
                    )
                )
            except Exception:
                pass

        # フォールバック：何もしない
        return base_system

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

        # length / mode
        length_mode = str(self.state.get("reply_length_mode") or self.llm_meta.get("reply_length_mode") or "auto")
        self.llm_meta["reply_length_mode"] = length_mode

        mode_current = str(judge_mode or self.state.get("judge_mode") or self.llm_meta.get("judge_mode") or "normal")
        self.llm_meta["judge_mode"] = mode_current
        self.state["judge_mode"] = mode_current
        self.judge_ai.set_mode(mode_current)

        # 0) persona の最新情報（失敗しても落とさない）
        try:
            persona_all = self.persona_ai.get_all(reload=True)
            self.llm_meta["persona"] = persona_all
            style_hint = persona_all.get("style_hint") or self.llm_meta.get("composer_style_hint", "")
            self.llm_meta["style_hint"] = style_hint
        except Exception as e:
            self.llm_meta["persona_error"] = str(e)

        # 1) Scene payload（world_state / scene_emotion）
        try:
            scene_payload = self.scene_ai.build_emotion_override_payload()
            self.llm_meta["world_state"] = scene_payload.get("world_state", {}) or {}
            self.llm_meta["scene_emotion"] = scene_payload.get("scene_emotion", {}) or {}
            self.llm_meta["world_error"] = None
        except Exception as e:
            self.llm_meta["world_error"] = str(e)
            self.llm_meta.setdefault("world_state", {})
            self.llm_meta.setdefault("scene_emotion", {})

        # 2) Mixer override（必ず llm_meta に載せる）
        try:
            emotion_override = self.mixer_ai.build_emotion_override() or {}
            if not isinstance(emotion_override, dict):
                emotion_override = {}
            self.llm_meta["emotion_override"] = emotion_override
        except Exception as e:
            self.llm_meta["emotion_override"] = {}
            self.llm_meta["emotion_override_error"] = str(e)
            if LYRA_DEBUG:
                st.error("[AnswerTalker] MixerAI failed")
                st.exception(e)

        # 3) system_prompt_used を確定して messages に適用
        system_prompt_used = self._build_system_prompt_used(
            messages=messages,
            emotion_override=self.llm_meta.get("emotion_override", {}) or {},
            mode_current=mode_current,
            length_mode=length_mode,
        )
        self.llm_meta["system_prompt_used"] = system_prompt_used or ""

        # persona側に replace_system_prompt があるなら使う
        try:
            if hasattr(self.persona, "replace_system_prompt"):
                messages_for_models = self.persona.replace_system_prompt(
                    messages=messages,
                    new_system_prompt=system_prompt_used,
                )
            else:
                messages_for_models = self._replace_system_prompt(messages, system_prompt_used)
        except Exception:
            messages_for_models = self._replace_system_prompt(messages, system_prompt_used)

        # 4) ModelsAI2.collect（★ここが “空のまま” が起きてるので絶対記録）
        try:
            results = self.models_ai.collect(
                messages_for_models,
                mode_current=mode_current,
                emotion_override=self.llm_meta.get("emotion_override", {}) or {},
                reply_length_mode=length_mode,
            )
            if not isinstance(results, dict):
                results = {}
            self.llm_meta["models"] = results
            self.llm_meta["models_error"] = None
        except Exception as e:
            self.llm_meta["models"] = {}
            self.llm_meta["models_error"] = str(e)
            self.llm_meta["models_trace"] = traceback.format_exc()
            if LYRA_DEBUG:
                st.error("[AnswerTalker] ModelsAI2.collect failed")
                st.exception(e)

            # ここで返す（ただし llm_meta は更新済み）
            self.state["llm_meta"] = self.llm_meta
            return "……ごめん、少し調子が悪いみたい。もう一度だけ言ってくれる？"

        # 5) Judge
        try:
            judge_result = self.judge_ai.run(
                self.llm_meta.get("models", {}) or {},
                user_text=user_text or "",
                preferred_length_mode=length_mode,
            )
            if not isinstance(judge_result, dict):
                judge_result = {"status": "error", "error": "judge_result is not dict"}
            self.llm_meta["judge"] = judge_result
            self.llm_meta["judge_error"] = None
        except Exception as e:
            self.llm_meta["judge"] = {
                "status": "error",
                "error": str(e),
                "chosen_model": "",
                "chosen_text": "",
                "candidates": [],
            }
            self.llm_meta["judge_error"] = str(e)
            if LYRA_DEBUG:
                st.error("[AnswerTalker] JudgeAI3 failed")
                st.exception(e)

        # 6) Composer
        try:
            composed = self.composer_ai.compose(self.llm_meta)
            if not isinstance(composed, dict):
                composed = {"status": "error", "error": "composer output is not dict", "text": ""}
            self.llm_meta["composer"] = composed
            self.llm_meta["composer_error"] = None
        except Exception as e:
            fallback = (self.llm_meta.get("judge", {}) or {}).get("chosen_text") or ""
            self.llm_meta["composer"] = {
                "status": "error",
                "error": str(e),
                "text": fallback,
                "source_model": (self.llm_meta.get("judge", {}) or {}).get("chosen_model", ""),
                "mode": "judge_fallback",
            }
            self.llm_meta["composer_error"] = str(e)
            if LYRA_DEBUG:
                st.error("[AnswerTalker] ComposerAI failed")
                st.exception(e)

        final_text = (self.llm_meta.get("composer", {}) or {}).get("text") or (self.llm_meta.get("judge", {}) or {}).get("chosen_text") or ""
        final_text = str(final_text)

        # 7) EmotionAI（失敗しても落とさない）
        try:
            emotion_res: EmotionResult = self.emotion_ai.analyze(
                composer=self.llm_meta.get("composer", {}) or {},
                memory_context=self.llm_meta.get("memory_context", "") or "",
                user_text=user_text or "",
            )
            emo_model = EmotionModel(result=emotion_res)
            emo_model.sync_relationship_fields()

            self.llm_meta["emotion"] = emotion_res.to_dict()
            next_mode = self.emotion_ai.decide_judge_mode(emotion_res)
            self.llm_meta["judge_mode_next"] = next_mode
            self.state["judge_mode"] = next_mode
        except Exception as e:
            self.llm_meta["emotion_error"] = str(e)
            if LYRA_DEBUG:
                st.error("[AnswerTalker] EmotionAI failed")
                st.exception(e)

        # 8) Memory 更新（ビューが見るので保存しておく）
        try:
            round_id = _safe_int(self.state.get("round_number", 0), 0)
            mem_update = self.memory_ai.update_from_turn(
                messages=messages_for_models,
                final_reply=final_text,
                round_id=round_id,
            )
            self.llm_meta["memory_update"] = mem_update
        except Exception as e:
            self.llm_meta["memory_update"] = {"status": "error", "error": str(e), "records": []}
            if LYRA_DEBUG:
                st.error("[AnswerTalker] MemoryAI.update_from_turn failed")
                st.exception(e)

        # ★ 最後に必ず state へ反映（ビューが読む）
        self.state["llm_meta"] = self.llm_meta

        return final_text
