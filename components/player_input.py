# components/player_input.py

from typing import List, Dict  # 使わなければ消してもOK
import streamlit as st


class PlayerInput:
    TEXT_KEY = "player_input_text"
    CLEAR_FLAG_KEY = "player_input_clear_flag"

    def __init__(self) -> None:
        # セッションの初期化だけ。ここでは heavy なことをしない
        if self.TEXT_KEY not in st.session_state:
            st.session_state[self.TEXT_KEY] = ""
        if self.CLEAR_FLAG_KEY not in st.session_state:
            st.session_state[self.CLEAR_FLAG_KEY] = False

    def render(self) -> str:
        """プレイヤーの入力欄を表示し、送信されたテキストを返す。"""

        # 前回送信後のクリアフラグが立っていたら、テキストエリアを空に戻す
        if st.session_state.get(self.CLEAR_FLAG_KEY, False):
            st.session_state[self.TEXT_KEY] = ""
            st.session_state[self.CLEAR_FLAG_KEY] = False

        st.markdown("**あなたの発言を入力：**")

        user_text: str = st.text_area(
            "",
            key=self.TEXT_KEY,
            height=160,
            label_visibility="collapsed",
        )

        submitted = st.button("送信", type="primary")

        if submitted:
            text_to_return = user_text.strip()
            st.session_state[self.CLEAR_FLAG_KEY] = True
            # デバッグしたければ↓一時的に有効化
            # st.text(f"DEBUG: {repr(text_to_return)}")
            return text_to_return

        return ""
