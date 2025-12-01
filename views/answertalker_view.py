# views/answertalker_view.py
from __future__ import annotations

from typing import Any, Dict, List, Protocol
import os
import json
import streamlit as st

from auth.roles import Role
from actors.actor import Actor
from actors.answer_talker import AnswerTalker
from personas.persona_floria_ja import Persona  # いまはフローリア固定


class View(Protocol):
    def render(self) -> None: ...


class AnswerTalkerView:
    """
    AnswerTalker / ModelsAI / JudgeAI3 / ComposerAI / MemoryAI の
    デバッグ・閲覧用ビュー。
    """

    TITLE = "🧩 AnswerTalker（AI統合テスト）"

    def __init__(self) -> None:
        persona = Persona()
        self.actor = Actor("floria", persona)

        debug_flag = os.getenv("LYRA_DEBUG", "").strip()
        if debug_flag == "1":
            # ★ Streamlit 側の state を AnswerTalker に明示的に渡す
            self.answer_talker = AnswerTalker(
                persona,
                state=st.session_state,
            )
        else:
            # ★ 本番モードなど、純粋な Python として使う場合
            self.answer_talker = AnswerTalker(persona)

    def render(self) -> None:
        st.header(self.TITLE)

        st.info(
            "この画面では、Actor に紐づく AnswerTalker が保持している llm_meta の内容 "
            "（models / judge / composer / emotion / memory）を参照できます。\n\n"
            "※ この画面からは AnswerTalker.run_models() や MemoryAI.update_from_turn() などは実行しません。"
        )

        llm_meta: Dict[str, Any] = st.session_state.get("llm_meta", {}) or {}

        # （以下、元の表示ロジックはそのまま）
        # ... ここはユーザーさんの最新版をそのまま使ってOK ...
