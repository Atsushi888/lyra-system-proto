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

    - ModelsAI で複数モデルから回答収集
    - JudgeAI2 で採用モデルを決定
    - ComposerAI で仕上げテキストを作る

    いまは最終返答として、まだ reply_text をそのまま返している。
    （後で composed["text"] に差し替える前提）
    """

    def __init__(self) -> None:
        # llm_meta をセッション状態から取得／初期化
        llm_meta = st.session_state.get("llm_meta")
        if not isinstance(llm_meta, dict):
            llm_meta = {
                "models": {},   # ModelsAI.collect の結果
                "judge": {},    # JudgeAI2.process の結果
                "composer": {}, # ComposerAI.compose の結果
            }

        self.llm_meta: Dict[str, Any] = llm_meta
        st.session_state["llm_meta"] = self.llm_meta

        # サブモジュール
        self.models_ai = ModelsAI()
        self.judge_ai = JudgeAI2()
        self.composer_ai = ComposerAI()

    # ===== モデル呼び出し入口 =====
    def run_models(self, messages: List[Dict[str, str]]) -> None:
        """
        Persona.build_messages で組み立てた messages を受け取り、
        ModelsAI.collect を通じて複数モデルから回答を収集する。
        """
        if not messages:
            return

        results = self.models_ai.collect(messages)
        self.llm_meta["models"] = results
        st.session_state["llm_meta"] = self.llm_meta

    # ===== パイプライン全体 =====
    def speak(
        self,
        reply_text: str,
        raw_result: Any | None = None,
        user_text: str = "",
        messages: List[Dict[str, str]] | None = None,
    ) -> str:
        """
        Actor から一次結果 reply_text と user_text / messages を受け取り、
        - ModelsAI.collect
        - JudgeAI2.process
        - ComposerAI.compose
        を順に実行する。

        いまのところ戻り値としては reply_text をそのまま返す。
        将来的には composed["text"] を返す想定。
        """
        if messages is None:
            messages = []

        # ① 複数モデルの回答収集
        self.run_models(messages)

        # ② JudgeAI2 で採択（失敗時も llm_meta にエラーを記録）
        try:
            judge_result = self.judge_ai.process(self.llm_meta)
        except Exception as e:
            judge_result = {
                "status": "error",
                "error": str(e),
            }
        self.llm_meta["judge"] = judge_result

        # ③ ComposerAI で仕上げ（失敗時は fallback として reply_text を保持）
        try:
            composed = self.composer_ai.compose(self.llm_meta)
        except Exception as e:
            composed = {
                "status": "error",
                "error": str(e),
                "text": reply_text,
            }
        self.llm_meta["composer"] = composed

        st.session_state["llm_meta"] = self.llm_meta

        # TODO: 将来ここを composed.get("text", reply_text) に切り替える
        return reply_text
