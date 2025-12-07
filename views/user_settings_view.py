# views/user_settings_view.py
from __future__ import annotations

from typing import Protocol

import streamlit as st

from components.user_settings import UserSettings


class View(Protocol):
    def render(self) -> None: ...


class UserSettingsView:
    """
    ユーザー設定（プレイヤー名・発話長さモードなど）を操作するためのビュー。
    ModeSwitcher から呼び出される前提。
    """

    TITLE = "⚙️ ユーザー設定（Player & Conversation）"

    def __init__(self) -> None:
        self.settings = UserSettings()

    def render(self) -> None:
        st.header(self.TITLE)
        st.caption(
            "プレイヤー名や、会話の長さモードなどをまとめて設定する画面です。\n"
            "今後、他のユーザー設定（難易度・表示言語など）もここに集約していく想定です。"
        )
        self.settings.render()


def create_user_settings_view() -> UserSettingsView:
    """
    ModeSwitcher から呼び出すためのファクトリ関数。
    """
        # typo防止のため単純にインスタンスを返すだけ
    return UserSettingsView()
