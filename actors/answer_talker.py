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
from actors.emotion.emotion_models import EmotionModel
from actors.persona_ai import PersonaAI
from actors.scene_ai import SceneAI
from actors.mixer_ai import MixerAI
from llm.llm_manager import LLMManager
from llm.llm_manager_factory import get_llm_manager
from actors.utils.debug_world_state import WorldStateDebugger  # 🔍 追加

# 環境変数でデバッグモードを切り替え
LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"

# このファイル専用 WorldState デバッガ
WS_DEBUGGER = WorldStateDebugger(name="AnswerTalker")


class AnswerTalker:
    """
    AI回答パイプラインの司令塔クラス。

    - ModelsAI2
    - JudgeAI3
    - ComposerAI
    - EmotionAI / EmotionModel
    - MemoryAI
    - PersonaAI（JSONベースの人格情報）
    - SceneAI / MixerAI（シーン＆感情オーバーライド）

    ※ system_prompt への感情ヘッダ付与や差し替えは、
       PersonaBase 側のメソッドに委譲する。
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
        llm_meta.setdefault("emotion_override", {})
        llm_meta.setdefault("system_prompt_used", {})
        llm_meta.setdefault("emotion_model_snapshot", {})
        self._ensure_world_state_controls(self.state)        
        
        # ★ 文章量モード（UserSettings 由来）
        length_mode = str(
            self.state.get("reply_length_mode")
            or llm_meta.get("reply_length_mode")
            or "auto"
        )
        llm_meta.setdefault("reply_length_mode", length_mode)
        self.state["reply_length_mode"] = length_mode

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

    def _ensure_world_state_controls(state):
        # --- world_state manual controls ---
        state.setdefault("world_state_manual_controls", {})
        mc = state["world_state_manual_controls"]
    
        if not isinstance(mc, dict):
            mc = {}
            state["world_state_manual_controls"] = mc
    
        mc.setdefault("others_present", False)
        mc.setdefault("interaction_mode_hint", "auto")  
        # ↑ narrator / scene / auto など、DokipowerControlで使う値


    # ---------------------------------------
    # ModelsAI 呼び出し
    # ---------------------------------------
    def run_models(
        self,
        messages: List[Dict[str, str]],
        mode_current: str = "normal",
        emotion_override: Optional[Dict[str, Any]] = None,
        *,
        length_mode: Optional[str] = None,
    ) -> None:
        if not messages:
            if LYRA_DEBUG:
                st.write("[DEBUG:AnswerTalker.run_models] messages is empty. skip.")
            return

        # length_mode が明示されなければ、llm_meta / state から拾う
        if not length_mode:
            length_mode = str(
                self.state.get("reply_length_mode")
                or self.llm_meta.get("reply_length_mode")
                or "auto"
            )

        if LYRA_DEBUG:
            st.write(
                f"[DEBUG:AnswerTalker.run_models] start: "
                f"len(messages)={len(messages)}, mode_current={mode_current}, "
                f"length_mode={length_mode}"
            )

        results = self.models_ai.collect(
            messages,
            mode_current=mode_current,
            emotion_override=emotion_override,
            reply_length_mode=length_mode,
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
            if LYRA_DEBUG:
                st.write(
                    "[DEBUG:AnswerTalker.speak] messages is empty → 空文字を返します。"
                )
            return ""

        if LYRA_DEBUG:
            st.write(
                "[DEBUG:AnswerTalker.speak] ========= TURN START ========="
            )
            st.write(
                f"[DEBUG:AnswerTalker.speak] len(messages)={len(messages)}, "
                f"user_text={repr(user_text)[:120]}, judge_mode={judge_mode}"
            )

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

        # 0.1) reply_length_mode 決定（UserSettings → llm_meta）
        length_mode = str(
            self.state.get("reply_length_mode")
            or self.llm_meta.get("reply_length_mode")
            or "auto"
        )
        self.llm_meta["reply_length_mode"] = length_mode

        if LYRA_DEBUG:
            st.write(f"[DEBUG:AnswerTalker.speak] judge_mode(current) = {mode_current}")
            st.write(f"[DEBUG:AnswerTalker.speak] reply_length_mode = {length_mode}")

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
            self.llm_meta["world_error"] = str(e)
            self.llm_meta.setdefault("world_state", {})
            self.llm_meta.setdefault("scene_emotion", {})

        # 🔍 デバッグ：SceneAI から積み上がった world_state / scene_emotion の確認
        WS_DEBUGGER.log(
            caller="AnswerTalker.speak[after_scene_payload]",
            world_state=self.llm_meta.get("world_state"),
            scene_emotion=self.llm_meta.get("scene_emotion"),
            extra={
                "step": "after_scene_payload",
                "has_world_error": bool(self.llm_meta.get("world_error")),
            },
        )

        # 1.5) emotion_override を MixerAI から取得
        emotion_override = self.mixer_ai.build_emotion_override()
        self.llm_meta["emotion_override"] = emotion_override or {}

        if LYRA_DEBUG:
            st.write(
                "[DEBUG:AnswerTalker.speak] emotion_override =",
                emotion_override,
            )

        # 🔍 デバッグ：MixerAI が返した override 一式
        if isinstance(emotion_override, dict):
            WS_DEBUGGER.log(
                caller="AnswerTalker.speak[after_mixer]",
                world_state=emotion_override.get("world_state"),
                scene_emotion=emotion_override.get("scene_emotion"),
                emotion=emotion_override.get("emotion"),
                extra={
                    "step": "after_mixer",
                    "has_emotion_override": True,
                },
            )
        else:
            WS_DEBUGGER.log(
                caller="AnswerTalker.speak[after_mixer]",
                extra={
                    "step": "after_mixer",
                    "has_emotion_override": False,
                    "emotion_override_type": str(type(emotion_override)),
                },
            )

        # 1.6) system_prompt に emotion_override ＋ length_mode を織り込む
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

        # PersonaBase 側のメソッドに完全委譲
        if hasattr(self.persona, "build_emotion_based_system_prompt"):
            system_prompt_used = self.persona.build_emotion_based_system_prompt(
                base_system_prompt=base_system_prompt,
                emotion_override=emotion_override,
                mode_current=mode_current,
                length_mode=length_mode,
            )
        else:
            # 念のためのフォールバック：何も加工しない
            system_prompt_used = base_system_prompt

        self.llm_meta["system_prompt_used"] = system_prompt_used

        # messages の system を差し替え
        if hasattr(self.persona, "replace_system_prompt"):
            messages_for_models = self.persona.replace_system_prompt(
                messages=messages,
                new_system_prompt=system_prompt_used,
            )
        else:
            # 互換性フォールバック：この場で簡易差し替え
            new_messages = list(messages)
            system_index = None
            for idx, m in enumerate(new_messages):
                if m.get("role") == "system":
                    system_index = idx
                    break

            system_message = {
                "role": "system",
                "content": system_prompt_used,
            }

            if system_index is not None:
                new_messages[system_index] = system_message
            else:
                new_messages.insert(0, system_message)

            messages_for_models = new_messages

        if LYRA_DEBUG:
            st.write(
                "[DEBUG:AnswerTalker.speak] system_prompt_used length =",
                len(system_prompt_used),
            )

        # 2) ModelsAI.collect
        self.run_models(
            messages_for_models,
            mode_current=mode_current,
            emotion_override=emotion_override,
            length_mode=length_mode,
        )

        # 3) JudgeAI3
        try:
            # UserSettings が session_state に書いた reply_length_mode を見る
            length_mode = str(self.state.get("reply_length_mode", "auto") or "auto")
            self.llm_meta["reply_length_mode"] = length_mode

            judge_result = self.judge_ai.run(
                self.llm_meta.get("models", {}),
                user_text=user_text or "",
                preferred_length_mode=length_mode,
            )
        except Exception as e:
            judge_result = {
                "status": "error",
                "error": str(e),
                "chosen_model": "",
                "chosen_text": "",
                "candidates": [],
            }
            self.llm_meta["judge"] = judge_result
        if LYRA_DEBUG:
            st.write(
                "[DEBUG:AnswerTalker.speak] judge_result.status =",
                judge_result.get("status"),
                ", chosen_model =",
                judge_result.get("chosen_model"),
                ", len(chosen_text) =",
                len(judge_result.get("chosen_text") or ""),
            )

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

        if LYRA_DEBUG:
            st.write(
                "[DEBUG:AnswerTalker.speak] composer.status =",
                composed.get("status"),
                ", mode =",
                composed.get("mode"),
                ", source_model =",
                composed.get("source_model"),
                ", len(text) =",
                len(composed.get("text") or ""),
            )

        # 5) EmotionAI.analyze + EmotionModel + decide_judge_mode
        try:
            emotion_res: EmotionResult = self.emotion_ai.analyze(
                composer=composed,
                memory_context=self.llm_meta.get("memory_context", ""),
                user_text=user_text or "",
            )

            # EmotionModel ラッパを介して関係フェーズを同期
            emo_model = EmotionModel(result=emotion_res)
            emo_model.sync_relationship_fields()

            # メタ情報保存（ビューで見れるように）
            self.llm_meta["emotion"] = emotion_res.to_dict()
            self.llm_meta["emotion_model_snapshot"] = emo_model.to_debug_snapshot()

            # judge_mode の決定（現状は従来ロジックを維持）
            next_mode = self.emotion_ai.decide_judge_mode(emotion_res)
            # 将来はこちらに寄せられる:
            # next_mode = emo_model.decide_judge_mode(current_mode=mode_current)

            self.llm_meta["judge_mode_next"] = next_mode
            self.state["judge_mode"] = next_mode

            if LYRA_DEBUG:
                st.write(
                    "[DEBUG:AnswerTalker.speak] EmotionResult.affection =",
                    emotion_res.affection,
                    ", with_doki =",
                    getattr(emotion_res, "affection_with_doki", emotion_res.affection),
                    ", doki_power =",
                    getattr(emotion_res, "doki_power", None),
                    ", doki_level =",
                    getattr(emotion_res, "doki_level", None),
                    ", relationship_stage =",
                    getattr(emotion_res, "relationship_stage", None),
                    ", relationship_label =",
                    getattr(emotion_res, "relationship_label", None),
                )
                st.write(
                    "[DEBUG:AnswerTalker.speak] emotion_model_snapshot =",
                    self.llm_meta.get("emotion_model_snapshot"),
                )
                st.write(
                    "[DEBUG:AnswerTalker.speak] judge_mode_next =",
                    next_mode,
                )

        except Exception as e:
            self.llm_meta["emotion_error"] = str(e)
            if LYRA_DEBUG:
                st.write("[DEBUG:AnswerTalker.speak] EmotionAI/EmotionModel error:", str(e))

        # 6) final text
        final_text = composed.get("text") or judge_result.get("chosen_text") or ""

        if LYRA_DEBUG:
            st.write(
                "[DEBUG:AnswerTalker.speak] final_text length =",
                len(final_text),
            )
            st.write(
                "[DEBUG:AnswerTalker.speak] final_text preview =",
                repr(final_text[:200]),
            )

        # 7) MemoryAI.update_from_turn
        try:
            round_val = int(self.state.get("round_number", 0))
            mem_update = self.memory_ai.update_from_turn(
                messages=messages_for_models,
                final_reply=final_text,
                round_id=round_val,
            )
        except Exception as e:
            mem_update = {
                "status": "error",
                "error": str(e),
                "records": [],
            }
            if LYRA_DEBUG:
                st.write("[DEBUG:AnswerTalker.speak] MemoryAI.update_from_turn error:", e)

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
            if LYRA_DEBUG:
                st.write(
                    "[DEBUG:AnswerTalker.speak] EmotionAI.update_long_term error:",
                    e,
                )

        # 8) 保存
        self.state["llm_meta"] = self.llm_meta

        if LYRA_DEBUG:
            st.write("[DEBUG:AnswerTalker.speak] ========= TURN END =========")

        return final_text
