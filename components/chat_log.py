# components/chat_log.py

from typing import List, Dict
import streamlit as st
import html


class ChatLog:
    def __init__(self, partner_name: str, display_limit: int = 20000):
        self.partner_name = partner_name
        self.display_limit = display_limit

        # CSSの注入
        st.markdown(
            """
            <style>
            /* 1人分の吹き出し全体（上下の間隔担当） */
            .chat-bubble-container {
                margin: 10px 0;          /* 吹き出し同士の間隔だけをここで管理 */
            }

            /* 吹き出し本体（内側の余白・枠・左上寄せ担当） */
            .chat-bubble {
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 6px 10px;       /* 上下左右の内側余白を控えめに */
                margin: 0;               /* 外側マージンはコンテナに任せる */
                background-color: #f9f9f9;
                white-space: pre-wrap;   /* 改行保持 */
                text-align: left;        /* 左寄せ */
                line-height: 1.55;
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

            # 吹き出しコンテナ＋本体をまとめて描画
            st.markdown(
                f"""
                <div class="chat-bubble-container">
                    <div class="chat-bubble {role_class}">
                        <b>{name}:</b><br>{safe_txt}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
