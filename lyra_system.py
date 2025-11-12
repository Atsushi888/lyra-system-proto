from __future__ import annotations
import streamlit as st
from auth.auth_manager import AuthManager
from auth.roles import Role
from components.mode_switcher import ModeSwitcher

# lyra_system.py（該当箇所だけ）
class LyraSystem:
    def __init__(self) -> None:
        st.set_page_config(page_title="Lyra System", layout="wide")
        self.auth = AuthManager()
        from components.mode_switcher import ModeSwitcher
        self.switcher = ModeSwitcher(default_key="PLAY", session_key="view_mode")

    def run(self) -> None:
        # ← location を明示して渡す（旧版でも try でフォールバックされる）
        res = self.auth.render_login(location="main")
        if res.status not in (True, "authenticated"):
            st.stop()

        from auth.roles import Role
        role = self.auth.role()
        self.switcher.render(user_role=role)

if __name__ == "__main__":
    LyraSystem().run()
