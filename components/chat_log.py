from typing import List, Dict
import streamlit as st
import html


class ChatLog:
    def __init__(self, partner_name: str, display_limit: int = 20000):
        self.partner_name = partner_name
        self.display_limit = display_limit

        st.markdown(
            """
            <style>
            .chat-bubble-container {
                margin: 10px 0;
            }

            .chat-bubble {
                border: 1px solid #ccc;
                border-radius: 8px;

                /* ← 上の余白を限界まで削る */
                padding: 0px 20px 8px 20px;   /* 上0, 右10, 下8, 左10 */

                margin: 0;
                background-color: #f9f9f9;
                white-space: pre-wrap;
                text-align: left;
                line-height: 1.5;
            }

            .chat-name {
                font-weight: bold;
                line-height: 0;   /* 1行目の高さを詰める */
                margin: 0;
                padding-top: 0px;   /* ほんの少しだけ余裕、もっと詰めたければ 0 に */
                display: inline-block;
            }

            .chat-bubble.assistant {
                background-color: #f2f2f2;
                border-color: #999;
            }
            .chat-bubble.user {
                background-color: #e8f2ff;
                border-color: #66aaff;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def render(self, messages: List[Dict[str, str]]) -> None:
        st.subheader("💬 会話ログ")

        if not messages:
            st.text("（まだ会話は始まっていません）")
            return

        for msg in messages[-self.display_limit:]:
            role = msg.get("role", "")
            txt = msg.get("content", "")

            if role == "assistant":
                name = self.partner_name
                role_class = "assistant"
            elif role == "user":
                name = "あなた"
                role_class = "user"
            else:
                name = role or "system"
                role_class = "assistant"

            safe_txt = html.escape(txt)

            st.markdown(
                f"""
                <div class="chat-bubble-container">
                    <div class="chat-bubble {role_class}">
                        <span class="chat-name">{name}:</span><br>{safe_txt}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
