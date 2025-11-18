# actors/answer_talker.py

from __future__ import annotations
from typing import Any, Dict, List
import streamlit as st

from actors.models_ai import ModelsAI
from actors.judge_ai2 import JudgeAI2
from actors.composer_ai import ComposerAI


class AnswerTalker:
    def __init__(self) -> None:
        # ここを「モデル定義の唯一のソース」にする
        self.model_props: Dict[str, Dict[str, Any]] = {
            "gpt4o": {
                "enabled": True,
                "priority": 3,             # Judge用
                "router_fn": "call_gpt4o", # ModelsAI用
                "label": "GPT-4o",
            },
            "gpt51": {
                "enabled": True,
                "priority": 2,
                "router_fn": "call_gpt51",
                "label": "GPT-5.1",
            },
            "hermes": {
                "enabled": True,
                "priority": 1,
                "router_fn": "call_hermes",
                "label": "Hermes 4",
            },
        }

        llm_meta = st.session_state.get("llm_meta")
        if not isinstance(llm_meta, dict):
            llm_meta = {
                "models": {},
                "judge": {},
                "composer": {},
            }

        self.llm_meta: Dict[str, Any] = llm_meta
        st.session_state["llm_meta"] = self.llm_meta

        # ★ model_props を子クラスに渡す
        self.models_ai = ModelsAI(self.model_props)
        self.judge_ai = JudgeAI2(self.model_props)
        self.composer_ai = ComposerAI()

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
        if messages is None:
            messages = []

        # Models
        self.run_models(messages)

        # # Judge
        # try:
        #     judge_result = self.judge_ai.process(self.llm_meta)
        # except Exception as e:
        #     judge_result = {"status": "error", "error": str(e)}
        # self.llm_meta["judge"] = judge_result

        # # Composer
        # try:
        #     composed = self.composer_ai.compose(self.llm_meta)
        # except Exception as e:
        #     composed = {"status": "error", "error": str(e), "text": reply_text}
        # self.llm_meta["composer"] = composed

        # st.session_state["llm_meta"] = self.llm_meta

        # # TODO: 後で composed["text"] に切り替える
        return reply_text
