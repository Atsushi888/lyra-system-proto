# actors/answer_talker.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from actors.models_ai import ModelsAI
from actors.judge_ai3 import JudgeAI3
from actors.composer_ai import ComposerAI
from actors.memory_ai import MemoryAI
from actors.emotion_ai import EmotionAI, EmotionResult  # ★ EmotionAI
from llm.llm_manager import LLMManager
from llm.llm_manager_factory import get_llm_manager


class AnswerTalker:
    """
    AI回答パイプラインの司令塔クラス。

    - ModelsAI:   複数モデルから回答収集（gpt4o / gpt51 / hermes / grok / gemini など）
    - JudgeAI3:   どのモデルの回答を採用するかを決定（モード切替対応）
    - ComposerAI: 採用候補をもとに最終的な返答テキストを生成
    - EmotionAI:  Composer の最終返答＋記憶コンテキストから感情値を推定
    - MemoryAI:   1ターンごとの会話から長期記憶を抽出・保存

    仕様:
      Persona と conversation_log から組み立てた messages を入力として、
      Models → Judge → Composer → Emotion → Memory → Emotion(long-term)
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
            llm_meta = {
                "models": {},
                "judge": {},
                "judge_mode": "normal",
                "composer": {},
                "emotion": {},
                "emotion_long_term": {},
                "memory_context": "",
                "memory_update": {},
            }

        self.llm_meta: Dict[str, Any] = llm_meta
        st.session_state["llm_meta"] = self.llm_meta

        # Multi-LLM 集計
        self.models_ai = ModelsAI(self.llm_manager)

        # JudgeAI3（初期モードは llm_meta["judge_mode"] または session_state）
        initial_mode = self.llm_meta.get(
            "judge_mode",
            st.session_state.get("judge_mode", "normal"),
        )
        self.judge_ai = JudgeAI3(mode=str(initial_mode))

        # Composer / Memory
        self.composer_ai = ComposerAI()
        self.memory_ai = MemoryAI(
            llm_manager=self.llm_manager,
            persona_id=persona_id,
            model_name=memory_model,
        )

        # EmotionAI（感情解析用。モデルは gpt-5.1 を想定）
        self.emotion_ai = EmotionAI(
            llm_manager=self.llm_manager,
            model_name="gpt51",
        )

    # ---------------------------------------
    # ModelsAI 呼び出し
    # ---------------------------------------
    def run_models(self, messages: List[Dict[str, str]]) -> None:
        """
        Persona.build_messages() で組み立てた messages を受け取り、
        ModelsAI.collect() を実行して llm_meta["models"] を更新する。
        """
        if not messages:
            return
        results = self.models_ai.collect(messages)
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
        """
        Actor から messages（および任意で user_text）を受け取り、
        - MemoryAI.build_memory_context
        - ModelsAI.collect
        - JudgeAI3.run
        - ComposerAI.compose
        - EmotionAI.analyze
        - MemoryAI.update_from_turn
        - EmotionAI.update_long_term / decide_judge_mode
        を順に実行して「最終返答テキスト」を返す。
        """
        if not messages:
            return ""

        # 0) Judge モード指定があれば反映
        if judge_mode is not None:
            try:
                self.judge_ai.set_mode(judge_mode)
                self.llm_meta["judge_mode"] = judge_mode
                st.session_state["judge_mode"] = judge_mode
            except Exception as e:
                self.llm_meta["judge_mode_error"] = str(e)

        # 1) 次ターン用の memory_context を構築
        try:
            mem_ctx = self.memory_ai.build_memory_context(user_query=user_text or "")
            self.llm_meta["memory_context_error"] = None
        except Exception as e:
            mem_ctx = ""
            self.llm_meta["memory_context_error"] = str(e)

        self.llm_meta["memory_context"] = mem_ctx

        # 2) 複数モデルの回答収集
        self.run_models(messages)

        # 3) JudgeAI3 による採択
        try:
            models = self.llm_meta.get("models", {})
            judge_result = self.judge_ai.run(models)
        except Exception as e:
            judge_result = {
                "status": "error",
                "error": str(e),
                "chosen_model": "",
                "chosen_text": "",
                "candidates": [],
            }

        self.llm_meta["judge"] = judge_result

        # 4) ComposerAI による仕上げ
        try:
            composed = self.composer_ai.compose(self.llm_meta)
        except Exception as e:
            fallback = ""
            if isinstance(judge_result, dict):
                fallback = judge_result.get("chosen_text") or ""
            composed = {
                "status": "error",
                "error": str(e),
                "text": fallback,
                "source_model": (
                    judge_result.get("chosen_model", "")
                    if isinstance(judge_result, dict)
                    else ""
                ),
                "mode": "judge_fallback",
            }

        self.llm_meta["composer"] = composed

        # 5) EmotionAI による感情解析（単発）
        try:
            if isinstance(composed, dict):
                emotion_result: EmotionResult = self.emotion_ai.analyze(
                    composer=composed,
                    memory_context=self.llm_meta.get("memory_context", ""),
                    user_text=user_text or "",
                )
                self.llm_meta["emotion"] = emotion_result.to_dict()
                self.llm_meta["emotion_error"] = None
            else:
                self.llm_meta["emotion_error"] = "composer is not a dict"
        except Exception as e:
            self.llm_meta["emotion_error"] = str(e)

        # 6) 最終返答テキスト決定
        final_text = ""
        if isinstance(composed, dict):
            final_text = composed.get("text") or ""
        if (not final_text) and isinstance(judge_result, dict):
            final_text = judge_result.get("chosen_text") or ""

        # 7) MemoryAI に記憶更新を依頼
        try:
            round_val_raw = st.session_state.get("round_number", 0)
            try:
                round_val = int(round_val_raw)
            except Exception:
                round_val = 0

            mem_update = self.memory_ai.update_from_turn(
                messages=messages,
                final_reply=final_text,
                round_id=round_val,
            )
        except Exception as e:
            mem_update = {
                "status": "error",
                "added": 0,
                "total": 0,
                "reason": "exception",
                "raw_reply": "",
                "records": [],
                "error": str(e),
            }

        self.llm_meta["memory_update"] = mem_update

        # 7.5) EmotionAI 長期感情の更新 ＋ 次回 judge_mode の決定
        try:
            new_records = mem_update.get("records") or []

            # 長期感情を更新
            long_term_state = self.emotion_ai.update_long_term(new_records)
            if hasattr(long_term_state, "to_dict"):
                self.llm_meta["emotion_long_term"] = long_term_state.to_dict()
            else:
                self.llm_meta["emotion_long_term"] = long_term_state

            # 次ターン用 judge_mode を決定
            next_mode = self.emotion_ai.decide_judge_mode()
            if next_mode:
                self.llm_meta["judge_mode_next"] = next_mode
                st.session_state["judge_mode"] = next_mode
        except Exception as e:
            self.llm_meta["emotion_long_term_error"] = str(e)

        # 8) llm_meta を保存
        st.session_state["llm_meta"] = self.llm_meta

        return final_text
