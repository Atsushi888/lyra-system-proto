# actors/answer_talker.py

from __future__ import annotations
from typing import Dict, Any
import streamlit as st

from actors.models_ai import ModelsAI
from actors.judge_ai2 import JudgeAI2
from actors.composer_ai import ComposerAI


class AnswerTalker:
    """
    AI回答パイプラインの司令塔。
    - ModelsAI で回答収集
    - JudgeAI2 で最適回答選別
    - ComposerAI で仕上げ
    """

    def __init__(self) -> None:
        llm_meta = st.session_state.get("llm_meta")

        if not isinstance(llm_meta, dict):
            llm_meta = {
                "models": {},
                "judge": {},
                "composer": {},
            }

        self.llm_meta: Dict[str, Any] = llm_meta
        st.session_state["llm_meta"] = self.llm_meta

        # ★ サブモジュール
        self.models_ai = ModelsAI()
        self.judge_ai = JudgeAI2()
        self.composer_ai = ComposerAI()

    # ★唯一のモデル呼び出しポイント
    def run_models(self, user_text: str) -> None:
        if not user_text:
            return

        results = self.models_ai.collect(user_text)
        self.llm_meta["models"] = results
        st.session_state["llm_meta"] = self.llm_meta

    def speak(self, reply_text: str, raw_result: Any | None = None, user_text: str = "") -> str:
        """
        Actor から渡された一次結果 reply_text を受け取り、
        - ModelsAI で多AI回答収集
        - JudgeAI2 で採択
        - ComposerAI で整形
        最終返答を返す。
        """

        # ① ModelsAI.collect
        self.run_models(user_text)

        # ② JudgeAI
        judge_result = self.judge_ai.process(self.llm_meta)
        self.llm_meta["judge"] = judge_result

        # ③ Composer
        composed = self.composer_ai.process(self.llm_meta)
        self.llm_meta["composer"] = composed

        st.session_state["llm_meta"] = self.llm_meta

        # ④ 最終返却
        # 将来的には composed["text"] を返す
        return reply_text
