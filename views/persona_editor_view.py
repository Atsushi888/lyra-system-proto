from __future__ import annotations

from typing import Protocol

import streamlit as st

from actors.persona.persona_editor import PersonaEditor


class View(Protocol):
    def render(self) -> None: ...


class PersonaEditorView:
    """
    PersonaEditor コンポーネントを表示するためだけの薄いラッパビュー。
    ModeSwitcher から呼び出される前提。
    """

    TITLE = "🖋️ キャラ設定（Persona JSON エディタ）"

    def __init__(self) -> None:
        self.editor = PersonaEditor()

    def render(self) -> None:
        st.header(self.TITLE)
        self.editor.render()


def create_persona_editor_view() -> PersonaEditorView:
    """
    ModeSwitcher から呼び出すためのファクトリ関数。
    """
    return PersonaEditorView()
