# lyra_system.py — AuthManagerを無効化し、強制ADMINで実行

from __future__ import annotations
import streamlit as st
from auth.roles import Role

# ModeSwitcher 読み込み（存在しない場合でも落ちないようにする）
try:
    from components.mode_switcher import ModeSwitcher
except Exception:
    ModeSwitcher = None  # type: ignore


class LyraSystem:
    """認証スキップ版：開発・検証専用"""

    def __init__(self) -> None:
        # ページ設定
        st.set_page_config(page_title="Lyra System", layout="wide")

        # 本来は AuthManager() だが、今はスキップ
        # from auth.auth_manager import AuthManager
        # self.auth = AuthManager()

        # ModeSwitcher は存在すれば使う
        self.switcher = ModeSwitcher(default_key="PLAY", session_key="view_mode") if ModeSwitcher else None

    def run(self) -> None:
        # ★認証スキップ：常に管理者ロールを返す
        role = Role.ADMIN

        # 表示
        st.markdown("<h1 style='text-align:center;'>🔓 Lyra System（Admin Bypass Mode）</h1>", unsafe_allow_html=True)
        st.caption("※ 認証をスキップして管理者権限で実行中。")

        # ModeSwitcher が存在すれば描画
        if self.switcher is not None:
            self.switcher.render(user_role=role)
        else:
            st.info("ModeSwitcher が見つかりません。仮画面を表示します。")
            st.write("Lyra System is running in Administrator bypass mode.")


if __name__ == "__main__":
    LyraSystem().run()
