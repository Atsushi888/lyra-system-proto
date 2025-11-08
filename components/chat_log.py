import html
import streamlit as st

class ChatLog:
    def __init__(self, partner_name: str, display_limit: int = 20000):
        self.partner_name = partner_name
        self.display_limit = display_limit

    def render(self, messages):
        st.subheader("💬 会話ログ")

        for msg in messages[-self.display_limit:]:
            role = msg.get("role", "")
            txt  = msg.get("content", "")
            css_class = "assistant" if role == "assistant" else "user"
            name = self.partner_name if role == "assistant" else "あなた"

            # HTML特殊文字をエスケープ（安全）
            safe_txt = html.escape(txt)

            # <pre> で改行維持、div で枠を適用
            html_block = f"""
            <div class="chat-bubble {css_class}">
                <b>{name}:</b><br>
                <pre style="margin:0; white-space:pre-wrap;">{safe_txt}</pre>
            </div>
            """
            st.markdown(html_block, unsafe_allow_html=True)
