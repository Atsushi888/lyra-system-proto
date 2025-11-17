# actors/answer_talker.py

from __future__ import annotations
from typing import Any, Dict, List
import streamlit as st

from actors.models_ai import ModelsAI
from actors.judge_ai2 import JudgeAI2
from actors.composer_ai import ComposerAI


class AnswerTalker:
    """
    AI回答パイプラインの司令塔。
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

        self.models_ai = ModelsAI()
        self.judge_ai = JudgeAI2()
        self.composer_ai = ComposerAI()

    # ★ ここは messages を受け取る
    def run_models(self, messages: List[Dict[str, str]]) -> None:
        if not messages:
            return

        results = self.models_ai.collect(messages)
        self.llm_meta["models"] = results
        st.session_state["llm_meta"] = self.llm_meta

    def speak(
        self,
        reply_text: str,
        raw_result: Any | None = None,
        user_text: str = "",
        messages: List[Dict[str, str]] | None = None,
    ) -> str:
        """
        Actor から一次結果 reply_text と user_text/messages を受け取り、
        Models → Judge → Composer を順に実行。
        """
        if messages is None:
            messages = []

        # ① Models
        self.run_models(messages)

        # ② Judge
        try:
            judge_result = self.judge_ai.process(self.llm_meta)
        except Exception as e:
            judge_result = {"status": "error", "error": str(e)}
        self.llm_meta["judge"] = judge_result

        # ③ Composer
        try:
            composed = self.composer_ai.compose(self.llm_meta)
        except Exception as e:
            composed = {"status": "error", "error": str(e), "text": reply_text}
        self.llm_meta["composer"] = composed

        st.session_state["llm_meta"] = self.llm_meta

        # 今は暫定で元の reply を返す
        return reply_text
