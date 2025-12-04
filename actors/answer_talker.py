# actors/answer_talker.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Mapping
import os

import streamlit as st

from actors.models_ai2 import ModelsAI2
from actors.judge_ai3 import JudgeAI3
from actors.composer_ai import ComposerAI
from actors.memory_ai import MemoryAI
from actors.emotion_ai import EmotionAI, EmotionResult
from actors.persona_ai import PersonaAI
from actors.scene_ai import SceneAI
from actors.mixer_ai import MixerAI
from llm.llm_manager import LLMManager
from llm.llm_manager_factory import get_llm_manager


class AnswerTalker:
    """
    AI回答パイプラインの司令塔クラス。

    - ModelsAI2
    - JudgeAI3
    - ComposerAI
    - EmotionAI
    - MemoryAI
    - PersonaAI（JSONベースの人格情報）
    - SceneAI / MixerAI（シーン＆感情オーバーライド）
    """

    def __init__(
        self,
        persona: Any,
        llm_manager: Optional[LLMManager] = None,
        memory_model: str = "gpt4o",
        state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.persona = persona
        persona_id = getattr(self.persona, "char_id", "default")

        # Streamlit あり／なし両対応の state
        env_debug = os.getenv("LYRA_DEBUG", "")
        if state is not None:
            # 明示的に渡された state を最優先
            self.state = state
        elif env_debug == "1":
            # デバッグ時は Streamlit の state を共有
            self.state = st.session_state
        else:
            # 現状は Streamlit 前提なので session_state を使う
            self.state = st.session_state

        # PersonaAI
        self.persona_ai = PersonaAI(persona_id=persona_id)

        # LLMManager
        self.llm_manager: LLMManager = llm_manager or get_llm_manager(persona_id)
        self.model_props: Dict[str, Dict[str, Any]] = self.llm_manager.get_model_props()

        # llm_meta 初期化
        llm_meta = self.state.get("llm_meta")
        if not isinstance(llm_meta, dict):
            llm_meta = {}

        llm_meta.setdefault("models", {})
        llm_meta.setdefault("judge", {})
        llm_meta.setdefault("judge_mode", "normal")
        llm_meta.setdefault("judge_mode_next", llm_meta.get("judge_mode", "normal"))
        llm_meta.setdefault("composer", {})
        llm_meta.setdefault("emotion", {})
        llm_meta.setdefault("memory_context", "")
        llm_meta.setdefault("memory_update", {})
        llm_meta.setdefault("emotion_long_term", {})

        # ★ シーン/ワールド情報用のスロットも確保
        llm_meta.setdefault("world_state", {})
        llm_meta.setdefault("scene_emotion", {})

        # Persona 由来のスタイルヒント（旧 Persona クラス経由のデフォルト）
        if "composer_style_hint" not in llm_meta:
            hint = ""
            if hasattr(self.persona, "get_composer_style_hint"):
                try:
                    hint = str(self.persona.get_composer_style_hint())
                except Exception:
                    hint = ""
            llm_meta["composer_style_hint"] = hint

        # ラウンド開始時の judge_mode リセット
        round_no_raw = self.state.get("round_number", 0)
        try:
            round_no = int(round_no_raw)
        except Exception:
            round_no = 0

        if round_no <= 1:
            llm_meta["judge_mode"] = "normal"
            llm_meta["judge_mode_next"] = "normal"
            self.state["judge_mode"] = "normal"
        else:
            if "judge_mode" in self.state:
                llm_meta["judge_mode"] = self.state["judge_mode"]
            else:
                self.state["judge_mode"] = llm_meta.get("judge_mode", "normal")

        self.llm_meta: Dict[str, Any] = llm_meta
        self.state["llm_meta"] = self.llm_meta

        # Multi-LLM 集計
        self.models_ai = ModelsAI2(llm_manager=self.llm_manager)

        # Emotion / Scene / Mixer
        self.emotion_ai = EmotionAI(
            llm_manager=self.llm_manager,
            model_name="gpt51",
        )
        self.scene_ai = SceneAI(state=self.state)
        self.mixer_ai = MixerAI(
            state=self.state,
            emotion_ai=self.emotion_ai,
            scene_ai=self.scene_ai,
        )

        # Judge / Composer / Memory
        initial_mode = (
            self.state.get("judge_mode")
            or self.llm_meta.get("judge_mode")
            or self.llm_meta.get("emotion", {}).get("mode")
            or "normal"
        )
        self.judge_ai = JudgeAI3(mode=str(initial_mode))

        self.composer_ai = ComposerAI(
            llm_manager=self.llm_manager,
            refine_model="gpt51",
        )
        self.memory_ai = MemoryAI(
            llm_manager=self.llm_manager,
            persona_id=persona_id,
            model_name=memory_model,
        )

    # ---------------------------------------
    # ModelsAI 呼び出し
    # ---------------------------------------
    def run_models(
        self,
        messages: List[Dict[str, str]],
        mode_current: str = "normal",
        emotion_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not messages:
            return

        results = self.models_ai.collect(
            messages,
            mode_current=mode_current,
            emotion_override=emotion_override,
        )
        self.llm_meta["models"] = results
        self.state["llm_meta"] = self.llm_meta

    # ---------------------------------------
    # メインパイプライン
    # ---------------------------------------
    def speak(
        self,
        messages: List[Dict[str, str]],
        user_text: str = "",
        judge_mode: Optional[str] = None,
    ) -> str:

        if not messages:
            return ""

        # 0.5) PersonaAI から最新 persona 情報を取得 → llm_meta へ
        try:
            persona_all = self.persona_ai.get_all(reload=True)
            self.llm_meta["persona"] = persona_all

            style_hint = (
                persona_all.get("style_hint")
                or self.llm_meta.get("composer_style_hint", "")
            )
            self.llm_meta["style_hint"] = style_hint
        except Exception as e:
            self.llm_meta["persona_error"] = str(e)

        # 0) judge_mode 決定
        mode_current = (
            judge_mode
            or self.llm_meta.get("judge_mode")
            or self.state.get("judge_mode")
            or "normal"
        )

        self.judge_ai.set_mode(mode_current)
        self.llm_meta["judge_mode"] = mode_current
        self.state["judge_mode"] = mode_current

        # 1) MemoryAI.build_memory_context
        try:
            mem_ctx = self.memory_ai.build_memory_context(user_query=user_text or "")
            self.llm_meta["memory_context"] = mem_ctx
            self.llm_meta["memory_context_error"] = None
        except Exception as e:
            self.llm_meta["memory_context_error"] = str(e)
            self.llm_meta["memory_context"] = ""

        # 1.2) SceneAI から world_state / scene_emotion を取得して llm_meta に積む
        try:
            scene_payload = self.scene_ai.build_emotion_override_payload()
            self.llm_meta["world_state"] = scene_payload.get("world_state", {})
            self.llm_meta["scene_emotion"] = scene_payload.get("scene_emotion", {})
            self.llm_meta["world_error"] = None
        except Exception as e:
            # 失敗しても致命的ではないのでエラーだけ残す
            self.llm_meta["world_error"] = str(e)
            self.llm_meta.setdefault("world_state", {})
            self.llm_meta.setdefault("scene_emotion", {})

        # 1.5) emotion_override を MixerAI から取得
        emotion_override = self.mixer_ai.build_emotion_override()

        # 2) ModelsAI.collect
        self.run_models(
            messages,
            mode_current=mode_current,
            emotion_override=emotion_override,
        )

        # 3) JudgeAI3
        try:
            judge_result = self.judge_ai.run(self.llm_meta.get("models", {}))
        except Exception as e:
            judge_result = {
                "status": "error",
                "error": str(e),
                "chosen_model": "",
                "chosen_text": "",
                "candidates": [],
            }
        self.llm_meta["judge"] = judge_result

        # 3.5) Composer 用 dev_force_model（開発中は Gemini 固定）
        self.llm_meta["dev_force_model"] = "gemini"

        # 4) ComposerAI
        try:
            composed = self.composer_ai.compose(self.llm_meta)
        except Exception as e:
            fallback = judge_result.get("chosen_text") or ""
            composed = {
                "status": "error",
                "error": str(e),
                "text": fallback,
                "source_model": judge_result.get("chosen_model", ""),
                "mode": "judge_fallback",
            }
        self.llm_meta["composer"] = composed

        # 5) EmotionAI.analyze + decide_judge_mode
        try:
            emotion_res: EmotionResult = self.emotion_ai.analyze(
                composer=composed,
                memory_context=self.llm_meta.get("memory_context", ""),
                user_text=user_text or "",
            )
            self.llm_meta["emotion"] = emotion_res.to_dict()

            next_mode = self.emotion_ai.decide_judge_mode(emotion_res)
            self.llm_meta["judge_mode_next"] = next_mode
            self.state["judge_mode"] = next_mode

        except Exception as e:
            self.llm_meta["emotion_error"] = str(e)

        # 6) final text
        final_text = composed.get("text") or judge_result.get("chosen_text") or ""

        # 7) MemoryAI.update_from_turn
        try:
            round_val = int(self.state.get("round_number", 0))
            mem_update = self.memory_ai.update_from_turn(
                messages=messages,
                final_reply=final_text,
                round_id=round_val,
            )
        except Exception as e:
            mem_update = {
                "status": "error",
                "error": str(e),
                "records": [],
            }
        self.llm_meta["memory_update"] = mem_update

        # 7.5) EmotionAI.update_long_term
        try:
            if hasattr(self.memory_ai, "get_all_records"):
                records = self.memory_ai.get_all_records()
            else:
                records = []

            lt_state = self.emotion_ai.update_long_term(
                memory_records=records,
                current_round=int(self.state.get("round_number", 0)),
                alpha=0.3,
            )

            if hasattr(lt_state, "to_dict"):
                self.llm_meta["emotion_long_term"] = lt_state.to_dict()
        except Exception as e:
            self.llm_meta["emotion_long_term_error"] = str(e)

        # 8) 保存
        self.state["llm_meta"] = self.llm_meta

        return final_text
