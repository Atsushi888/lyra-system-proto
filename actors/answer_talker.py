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

# 環境変数でデバッグモードを切り替え
LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"


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
        llm_meta.setdefault("emotion_override", {})
        llm_meta.setdefault("system_prompt_used", "")

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
            if LYRA_DEBUG:
                st.write("[DEBUG:AnswerTalker.run_models] messages is empty. skip.")
            return

        if LYRA_DEBUG:
            st.write(
                f"[DEBUG:AnswerTalker.run_models] start: "
                f"len(messages)={len(messages)}, mode_current={mode_current}"
            )

        results = self.models_ai.collect(
            messages,
            mode_current=mode_current,
            emotion_override=emotion_override,
        )
        self.llm_meta["models"] = results
        self.state["llm_meta"] = self.llm_meta

    # ---------------------------------------
    # system_prompt に emotion_override を反映
    # ---------------------------------------
    def _inject_emotion_into_system_prompt(
        self,
        messages: List[Dict[str, str]],
        mode_current: str,
        emotion_override: Optional[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """
        - 既存 messages から system prompt を探す
        - Persona 情報 + emotion_override で上書きした system_prompt を生成
        - messages の先頭 system を差し替え（なければ先頭に追加）
        - 生成した system_prompt を llm_meta["system_prompt_used"] に保存
        """

        if emotion_override is None:
            emotion_override = {}

        # 既存 system prompt を取得
        base_system_prompt = ""
        system_index = None
        for idx, m in enumerate(messages):
            if m.get("role") == "system":
                base_system_prompt = m.get("content", "") or ""
                system_index = idx
                break

        # Persona 側からデフォルトの system_prompt が取れるなら、そちらも候補に
        if not base_system_prompt and hasattr(self.persona, "get_system_prompt"):
            try:
                base_system_prompt = str(self.persona.get_system_prompt())
            except Exception:
                pass

        # なければ空からでも良い
        world_state = emotion_override.get("world_state") or {}
        scene_emotion = emotion_override.get("scene_emotion") or {}
        emotion = emotion_override.get("emotion") or {}

        # affection は doki 補正後を優先
        affection = float(emotion.get("affection_with_doki", emotion.get("affection", 0.0)))
        doki_power = int(emotion.get("doki_power", 0) or 0)
        doki_level = int(emotion.get("doki_level", 0) or 0)

        # ザックリとしたラベル
        if affection >= 0.9:
            aff_label = "強い恋心（ほぼ告白寸前のレベル）"
        elif affection >= 0.7:
            aff_label = "はっきりとした好意（明らかに特別扱いしているレベル）"
        elif affection >= 0.4:
            aff_label = "穏やかな好意（友好的〜少し気になっている程度）"
        else:
            aff_label = "まだ大きな感情変化はない（普通〜やや好意寄り）"

        # ドキドキの説明
        if doki_level >= 3:
            doki_label = "鼓動が速く、視線や距離感に敏感になっている高揚状態"
        elif doki_level == 2:
            doki_label = "少し緊張しつつも、相手を意識してしまう状態"
        elif doki_level == 1:
            doki_label = "心のどこかで相手を気にしている軽いときめき"
        else:
            doki_label = "特別な高鳴りはないが、落ち着いた会話ができる状態"

        # 場所と時間はあれば一応説明に含める
        loc_player = world_state.get("locations", {}).get("player")
        time_slot = world_state.get("time", {}).get("slot")
        time_str = world_state.get("time", {}).get("time_str")

        location_hint_lines: List[str] = []
        if loc_player:
            location_hint_lines.append(f"- 現在の舞台は「{loc_player}」です。")
        if time_slot or time_str:
            ts = f"{time_slot} / {time_str}" if time_slot and time_str else (time_slot or time_str)
            location_hint_lines.append(f"- 時間帯は「{ts}」です。")

        location_hint = "\n".join(location_hint_lines)

        # override 指示文
        override_lines = [
            "[Lyra-System Emotion Override 指示]",
            f"- 現在の感情モード: {mode_current}",
            f"- 好感レベル (affection_with_doki): {affection:.2f}",
            f"  → 解釈: {aff_label}",
            f"- 一時的なドキドキ状態 (doki_power: {doki_power}, doki_level: {doki_level})",
            f"  → 解釈: {doki_label}",
        ]
        if location_hint:
            override_lines.append(location_hint)

        override_lines.append(
            "- 上記の好意・ドキドキ状態をセリフのトーン・距離感・表情描写に必ず反映してください。"
        )
        override_lines.append(
            "- ただしキャラクター設定と世界観から逸脱しない範囲で、"
            "『今はかなりデレ寄り』であることが分かるように、"
            "行間や仕草・言い回しでしっかり表現してください。"
        )

        # mode が erotic の場合は、表現方針だけ少し追記（R18には踏み込まない）
        if str(mode_current) == "erotic":
            override_lines.append(
                "- mode=erotic の場合も、直接的な描写は避け、"
                "甘いロマンス寄りの表現や、少し大胆なスキンシップの匂わせまでに留めてください。"
            )

        override_block = "\n".join(override_lines)

        if base_system_prompt:
            new_system_prompt = base_system_prompt.rstrip() + "\n\n" + override_block + "\n"
        else:
            new_system_prompt = override_block + "\n"

        # llm_meta に保存（AnswerTalkerView でそのまま表示される想定）
        self.llm_meta["system_prompt_used"] = new_system_prompt

        # messages の先頭 system を差し替え / 追加
        new_messages = list(messages)
        if system_index is not None:
            new_messages[system_index] = {
                "role": "system",
                "content": new_system_prompt,
            }
        else:
            new_messages.insert(
                0,
                {
                    "role": "system",
                    "content": new_system_prompt,
                },
            )

        if LYRA_DEBUG:
            st.write(
                "[DEBUG:AnswerTalker._inject_emotion_into_system_prompt] "
                f"system_index={system_index}, len(new_system_prompt)={len(new_system_prompt)}"
            )

        return new_messages

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

        if LYRA_DEBUG:
            st.write(f"[DEBUG:AnswerTalker.speak] judge_mode(current) = {mode_current}")

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

        # 1.5) emotion_override を MixerAI から取得
        emotion_override = self.mixer_ai.build_emotion_override()
        self.llm_meta["emotion_override"] = emotion_override or {}

        if LYRA_DEBUG:
            st.write(
                "[DEBUG:AnswerTalker.speak] emotion_override =",
                emotion_override,
            )

        # 1.6) system_prompt に emotion_override を織り込む
        messages_for_models = self._inject_emotion_into_system_prompt(
            messages=messages,
            mode_current=mode_current,
            emotion_override=emotion_override,
        )

        # 2) ModelsAI.collect
        self.run_models(
            messages_for_models,
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

        if LYRA_DEBUG:
            st.write(
                "[DEBUG:AnswerTalker.speak] judge_result.status =",
                judge_result.get("status"),
                ", chosen_model =",
                judge_result.get("chosen_model"),
                ", len(chosen_text) =",
                len(judge_result.get("chosen_text") or ""),
            )

        # 3.5) Composer 用 dev_force_model（開発中は Gemini 固定も可）
        # self.llm_meta["dev_force_model"] = "gemini"

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
                )
                st.write(
                    "[DEBUG:AnswerTalker.speak] judge_mode_next =",
                    next_mode,
                )

        except Exception as e:
            self.llm_meta["emotion_error"] = str(e)
            if LYRA_DEBUG:
                st.write("[DEBUG:AnswerTalker.speak] EmotionAI error:", str(e))

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
