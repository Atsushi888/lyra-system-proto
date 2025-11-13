# views/user_view.py
import os
# from __future__ import annotations
import streamlit as st
from components import PreflightChecker

class UserView:
    def __init__( self ):
        # APIキー
        openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        openrouter_key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))

        if not openai_key:
            st.error("OPENAI_API_KEY が未設定です。Settings → Secrets で設定してください。")
            st.stop()
        os.environ["OPENAI_API_KEY"] = openai_key
        if openrouter_key:
            os.environ["OPENROUTER_API_KEY"] = openrouter_key        
        self.preflight  = PreflightChecker(openai_key, openrouter_key)

def render(self) -> None:
    st.caption("公開向けの軽量設定のみを表示")

    # 上段：preflight
    with st.container():
        self.preflight.render()

    st.markdown("---")  # 仕切り線はお好みで

    # 下段：トグル群
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.toggle(
                "字幕を表示",
                key="ui_show_subtitle",
                value=st.session_state.get("ui_show_subtitle", True),
            )
        with col2:
            st.toggle(
                "効果音を有効にする",
                key="ui_sfx",
                value=st.session_state.get("ui_sfx", True),
            )
