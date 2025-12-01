# views/scene_changer_view.py
from __future__ import annotations

from typing import Protocol

import streamlit as st

from components.scene_changer import SceneChanger


class View(Protocol):
    def render(self) -> None: ...


class SceneChangerView:
    """
    SceneChanger コンポーネントを表示するためだけの薄いラッパビュー。
    ModeSwitcher から呼び出される前提。
    """

    TITLE = "🚶‍♀️ シーン移動"

    def __init__(self) -> None:
        self.changer = SceneChanger()

    def render(self) -> None:
        st.header(self.TITLE)
        self.changer.render()


def create_scene_changer_view() -> SceneChangerView:
    """
    ModeSwitcher から呼び出すためのファクトリ関数。
    """
    return SceneChangerView()
