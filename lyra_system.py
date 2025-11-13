# lyra_system.py — Lyra 全体を束ねるエントリポイント（認証バイパス開発版）

from __future__ import annotations

import streamlit as st

from auth.roles import Role
from components.mode_switcher import ModeSwitcher


class LyraSystem:
    """
    開発用エントリポイント。

    - 認証(AuthManager)は **いったん完全バイパス** して、常に ADMIN 権限として扱う。
    - 画面切り替え（4ボタン）は ModeSwitcher に全委譲。
    - 認証周りが安定したら、ここに AuthManager 呼び出しを復活させる。
    """

    def __init__(self) -> None:
        # ページ全体設定
        st.set_page_config(
            page_title="Lyra System",
            layout="wide",
        )

        # ★ 認証は当面使わないので、AuthManager は封印
        # from auth.auth_manager import AuthManager
        # self.auth = AuthManager()

        # 画面切り替えコントローラ
        self.switcher = ModeSwitcher(
            default_key="PLAY",
            session_key="view_mode",
        )

    def run(self) -> None:
        # ============================
        #  開発モード：常に ADMIN 扱い
        # ============================
        role = Role.ADMIN
        # role = Role.USER

        # サイドバーに「開発中・認証バイパス中」の注意を出しておく
        # with st.sidebar:
            # st.markdown("### 画面切替")
            # st.caption("※ 現在は **認証バイパス中（開発モード）** です。")

        # 画面切り替え本体を実行
        self.switcher.render(user_role=role)


if __name__ == "__main__":
    LyraSystem().run()
