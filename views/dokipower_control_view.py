# views/dokipower_control_view.py
from __future__ import annotations

from typing import Protocol

import streamlit as st

from components.dokipower_control import DokiPowerController


class View(Protocol):
    def render(self) -> None: ...


class DokiPowerControlView:
    """
    DokiPowerController コンポーネントを表示するためだけの薄いラッパビュー。
    ModeSwitcher から呼び出される前提。
    """

    TITLE = "💓 ドキドキパワー調整（Emotion Debug）"

    def __init__(self) -> None:
        self.controller = DokiPowerController()

    def render(self) -> None:
        st.header(self.TITLE)
        st.caption(
            "ドキドキ💓パワーと感情値を手動で調整し、"
            "MixerAI などから参照するためのデバッグ用パネルです。"
        )
        self.controller.render()


def create_dokipower_control_view() -> DokiPowerControlView:
    """
    ModeSwitcher から呼び出すためのファクトリ関数。
    """
    return DokiPowerControlView()
