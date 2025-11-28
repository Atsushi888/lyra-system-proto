# actors/answer_talker.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from actors.models_ai2 import ModelsAI2
from actors.judge_ai3 import JudgeAI3
from actors.composer_ai import ComposerAI
from actors.memory_ai import MemoryAI
from actors.emotion_ai import EmotionAI, EmotionResult
from llm.llm_manager import LLMManager
from llm.llm_manager_factory import get_llm_manager


class AnswerTalker:
    """
    AI回答パイプラインの司令塔クラス。

    - ModelsAI2:  複数モデルから回答収集（gpt51 / grok / gemini / hermes / ...）
    - JudgeAI3:   どのモデルの回答を採用するかを決定（モード切替対応）
    - ComposerAI: 採用候補をもとに最終的な返答テキストを生成（＋将来Refine）
    - EmotionAI:  Composer の最終返答＋記憶コンテキストから感情値を推定
                  ＋ 長期感情（LongTermEmotion）の更新と judge_mode 決定
    - MemoryAI:   1ターンごとの会話から長期記憶を抽出・保存

    仕様:
      Persona と conversation_log から組み立てた messages を入力として、
      Models → Judge → Composer → Emotion(short) → Memory → Emotion(long)
      を回し、最終的な返答テキストを返す。
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

        # 登録されているモデルメタ情報
        self.model_props: Dict[str, Dict[str, Any]] = self.llm_manager.get_model_props()

        # llm_meta の初期化／復元
        llm_meta = st.session_state.get("llm_meta")
        if not isinstance(llm_meta, dict):
            llm_meta = {}

        # 必要キーのデフォルトを埋める
        llm_meta.setdefault("models", {})
        llm_meta.setdefault("judge", {})
        llm_meta.setdefault("judge_mode", "normal")  # このターンで実際に使ったモード
        llm_meta.setdefault("judge_mode_next", llm_meta.get("judge_mode", "normal"))
        llm_meta.setdefault("composer", {})
        llm_meta.setdefault("emotion", {})
        llm_meta.setdefault("memory_context", "")
        llm_meta.setdefault("memory_update", {})
        llm_meta.setdefault("emotion_long_term", {})

        # ★ Persona 由来のスタイルヒントを llm_meta に載せておく
        #    Refiner や将来のライティング系で使うための導線だけ確保
        if "composer_style_hint" not in llm_meta:
            hint = ""
            if hasattr(self.persona, "get_composer_style_hint"):
                try:
                    hint = str(self.persona.get_composer_style_hint())
                except Exception:
                    hint = ""
            llm_meta["composer_style_hint"] = hint

        # ----- ★ 新しい会談開始時は強制 normal にリセットする -----
        round_no_raw = st.session_state.get("round_number", 0)
        try:
            round_no = int(round_no_raw)
        except Exception:
            round_no = 0

        if round_no <= 1:
            # ラウンド 0〜1 は「新規会談」とみなし、モードをリセット
            llm_meta["judge_mode"] = "normal"
            llm_meta["judge_mode_next"] = "normal"
            st.session_state["judge_mode"] = "normal"
        else:
            # 既存会談では session_state 優先で llm_meta を上書き
            if "judge_mode" in st.session_state:
                llm_meta["judge_mode"] = st.session_state["judge_mode"]
            else:
                # 念のため同期
                st.session_state["judge_mode"] = llm_meta.get("judge_mode", "normal")

        self.llm_meta: Dict[str, Any] = llm_meta
        st.session_state["llm_meta"] = self.llm_meta

        # Multi-LLM 集計（新バージョン）
        self.models_ai = ModelsAI2(llm_manager=self.llm_manager)

        # EmotionAI（感情解析＋長期感情管理）
        self.emotion_ai = EmotionAI(
            llm_manager=self.llm_manager,
            model_name="gpt51",
        )

        # JudgeAI3（初期モードは llm_meta / session_state / emotion のいずれか）
        initial_mode = (
            st.session_state.get("judge_mode")
            or self.llm_meta.get("judge_mode")
            or self.llm_meta.get("emotion", {}).get("mode")
            or "normal"
        )
        self.judge_ai = JudgeAI3(mode=str(initial_mode))

        # Composer / Memory
        # ★ ここで llm_manager を渡しておくことで Refiner を有効にできる
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
    # emotion_override 構築ヘルパ
    # ---------------------------------------
    def _build_emotion_override_for_models(self) -> Optional[Dict[str, Any]]:
        """
        ModelsAI2.collect() に渡す emotion_override を構築する。

        emotion_override_mode:
          - "auto"        : EmotionAI の短期感情をそのまま渡す
          - "manual_full" : 手動パネルの値で完全上書き（EmotionAI は無視）
        """

        mode = st.session_state.get("emotion_override_mode", "auto")
        manual = st.session_state.get("emotion_override_manual")

        # ---------------------------
        # 1) 手動完全上書きモード
        # ---------------------------
        if mode == "manual_full":
            if isinstance(manual, dict):
                return {
                    "mode": manual.get("mode", "normal"),
                    "affection": float(manual.get("affection", 0.0)),
                    "arousal": float(manual.get("arousal", 0.0)),
                    "tension": float(manual.get("tension", 0.0)),
                    "anger": float(manual.get("anger", 0.0)),
                    "sadness": float(manual.get("sadness", 0.0)),
                    "excitement": float(manual.get("excitement", 0.0)),
                }
            return None

        # ---------------------------
        # 2) 通常（auto）モード
        # ---------------------------
        if not hasattr(self, "emotion_ai") or self.emotion_ai is None:
            return None

        short: Optional[EmotionResult] = getattr(
            self.emotion_ai, "last_short_result", None
        )
        if short is None:
            return None

        return {
            "mode": short.mode or "normal",
            "affection": float(short.affection),
            "arousal": float(short.arousal),
            "tension": float(short.tension),
            "anger": float(short.anger),
            "sadness": float(short.sadness),
            "excitement": float(short.excitement),
        }

    # ---------------------------------------
    # ModelsAI 呼び出し
    # ---------------------------------------
    def run_models(
        self,
        messages: List[Dict[str, str]],
        mode_current: str = "normal",
        emotion_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Persona.build_messages() で組み立てた messages を受け取り、
        ModelsAI2.collect() を実行して llm_meta["models"] を更新する。

        mode_current:
            "normal" / "erotic" / "debate" など、現在の Judge モード。
        """
        if not messages:
            return

        results = self.models_ai.collect(
            messages,
            mode_current=mode_current,
            emotion_override=emotion_override,
        )
        self.llm_meta["models"] = results
        st.session_state["llm_meta"] = self.llm_meta

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

        # ======================================================
        # 0) 「このターンで使う judge_mode」を決定
        # ======================================================
        mode_current = (
            judge_mode
            or self.llm_meta.get("judge_mode")
            or st.session_state.get("judge_mode")
            or "normal"
        )

        self.judge_ai.set_mode(mode_current)
        self.llm_meta["judge_mode"] = mode_current
        st.session_state["judge_mode"] = mode_current

        # ======================================================
        # 1) MemoryAI.build_memory_context
        # ======================================================
        try:
            mem_ctx = self.memory_ai.build_memory_context(user_query=user_text or "")
            self.llm_meta["memory_context"] = mem_ctx
            self.llm_meta["memory_context_error"] = None
        except Exception as e:
            self.llm_meta["memory_context_error"] = str(e)
            self.llm_meta["memory_context"] = ""

        # ======================================================
        # 1.5) emotion_override を構築（auto / manual_full）
        # ======================================================
        emotion_override = self._build_emotion_override_for_models()

        # ======================================================
        # 2) ModelsAI.collect
        # ======================================================
        self.run_models(
            messages,
            mode_current=mode_current,
            emotion_override=emotion_override,
        )

        # ======================================================
        # 3) JudgeAI3
        # ======================================================
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

        # ======================================================
        # 4) ComposerAI
        # ======================================================
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

        # ======================================================
        # 5) EmotionAI.analyze + decide_judge_mode
        # ======================================================
        try:
            emotion_res: EmotionResult = self.emotion_ai.analyze(
                composer=composed,
                memory_context=self.llm_meta.get("memory_context", ""),
                user_text=user_text or "",
            )
            self.llm_meta["emotion"] = emotion_res.to_dict()

            # 次ターンの judge_mode_next を EmotionAI に委譲
            next_mode = self.emotion_ai.decide_judge_mode(emotion_res)
            self.llm_meta["judge_mode_next"] = next_mode
            st.session_state["judge_mode"] = next_mode

        except Exception as e:
            self.llm_meta["emotion_error"] = str(e)

        # ======================================================
        # 6) 最終返答テキスト
        # ======================================================
        final_text = composed.get("text") or judge_result.get("chosen_text") or ""

        # ======================================================
        # 7) MemoryAI.update_from_turn
        # ======================================================
        try:
            round_val = int(st.session_state.get("round_number", 0))
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

        # ======================================================
        # 7.5) EmotionAI.update_long_term
        # ======================================================
        try:
            if hasattr(self.memory_ai, "get_all_records"):
                records = self.memory_ai.get_all_records()
            else:
                records = []

            lt_state = self.emotion_ai.update_long_term(
                memory_records=records,
                current_round=round_val,
                alpha=0.3,
            )

            if hasattr(lt_state, "to_dict"):
                self.llm_meta["emotion_long_term"] = lt_state.to_dict()
        except Exception as e:
            self.llm_meta["emotion_long_term_error"] = str(e)

        # ======================================================
        # 8) 保存
        # ======================================================
        st.session_state["llm_meta"] = self.llm_meta

        return final_text
