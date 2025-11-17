# actors/answer_talker.py

from __future__ import annotations
from typing import Dict, Any

import streamlit as st

from actors.models_ai import ModelsAI   # ← 修正ポイント


class AnswerTalker:
    """
    回答生成パイプラインの中核ベースクラス。

    現段階：
      - llm_meta の器だけ保持
      - speak は reply_text をそのまま返す
      - ModelsAI から「AIごとの回答一覧」を集める
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

        # Models → ModelsAI に変更
        self.models_ai = ModelsAI()

    def run_models(self, user_text: str) -> None:
        """ModelsAI を走らせて llm_meta['models'] を更新する"""

        if not user_text:
            return

        results = self.models_ai.collect(user_text)

        if not isinstance(self.llm_meta.get("models"), dict):
            self.llm_meta["models"] = {}

        self.llm_meta["models"] = results
        st.session_state["llm_meta"] = self.llm_meta

    def speak(self, reply_text: str, raw_result: Any | None = None) -> str:
        """今はまだ reply_text をそのまま返すだけ"""
        return reply_text
