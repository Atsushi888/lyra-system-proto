# views/ai_manager_view.py
from __future__ import annotations

from typing import Any
import streamlit as st

from components.ai_manager import AIManager


class AIManagerView:
    """
    AIManager.render() を呼ぶだけの薄いラッパ。
    """
    TITLE = "🤖 AI Manager"

    def __init__(self) -> None:
        # persona_id は今の系だと default で回してOK（必要なら後で差し込む）
        self.mgr = AIManager(persona_id="default")

    def render(self) -> None:
        self.mgr.render()


def create_ai_manager_view() -> AIManagerView:
    return AIManagerView()
