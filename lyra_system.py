# lyra_system.py — Lyra 全体を束ねるエントリポイント（認証バイパス開発版）

from __future__ import annotations

import importlib
import traceback

import streamlit as st


def safe_import(module_name: str):
    """
    Streamlit Cloud でログが握り潰される/見えないケースに備えて、
    import 失敗時の traceback を UI に強制表示する。
    """
    try:
        return importlib.import_module(module_name)
    except Exception:
        st.error(f"❌ import failed: {module_name}")
        st.code(traceback.format_exc())
        st.stop()


# ===== ここを safe_import 経由にして、落ちた地点を確実に炙り出す =====
roles_mod = safe_import("auth.roles")
Role = roles_mod.Role

ms_mod = safe_import("components.mode_switcher")
ModeSwitcher = ms_mod.ModeSwitcher


class LyraSystem:
    """
    開発用エントリポイント。

    - 認証(AuthManager)は **いったん完全バイパス** して、常に ADMIN 権限として扱う。
    - 画面切り替え（4ボタン）は ModeSwitcher に全委譲。
    - 認証周りが安定したら、ここに AuthManager 呼び出しを復活させる。
    """

    def __init__(self) -> None:
        # ページ全体設定（※ Streamlit は set_page_config を最初に1回だけ推奨）
        st.set_page_config(
            page_title="Lyra System",
            layout="wide",
        )

        # ★ 認証は当面使わないので、AuthManager は封印
        # from auth.auth_manager import AuthManager
        # self.auth = AuthManager()

        # 画面切り替えコントローラ
        self.switcher = ModeSwitcher(
            default_key="COUNCIL",
            session_key="view_mode",
        )

    def run(self) -> None:
        # ============================
        #  開発モード：常に ADMIN 扱い
        # ============================
        role = Role.ADMIN
        # role = Role.USER

        # 画面切り替え本体を実行（実行時例外も UI に吐く）
        try:
            self.switcher.render(user_role=role)
        except Exception:
            st.error("❌ runtime error in ModeSwitcher.render()")
            st.code(traceback.format_exc())
            st.stop()


if __name__ == "__main__":
    LyraSystem().run()
