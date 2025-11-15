# views/council_view.py

from __future__ import annotations
import streamlit as st

from council.council_manager import CouncilManager


class CouncilView:
    """
    Council Prototype (β) の UI 表示。
    """

    def __init__(self):
        self.manager = CouncilManager()

    def render(self):
        st.header("💬 会談システム（β）")

        st.subheader("会談ログ")
        for idx, msg in enumerate(self.manager.conversation_log):
            st.markdown(msg["content"], unsafe_allow_html=True)
        st.subheader("プレイヤー入力")
        user_text = st.text_area("あなたの発言:", "")

        if st.button("送信"):
            if user_text.strip():
                ai_reply = self.manager.proceed(user_text.strip())
                st.experimental_rerun()

        if st.button("会談リセット / 開始"):
            self.manager.reset()
            st.experimental_rerun()
