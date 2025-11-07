# components/player_input.py
import streamlit as st


class PlayerInput:
    """ユーザーの入力欄＋送信処理を担当"""

    def __init__(self, key_input: str = "user_input_box", key_button: str = "send_btn"):
        self.key_input = key_input
        self.key_button = key_button

    def render(self) -> str:
        """入力欄を描画して、送信ボタンが押されたときだけ文字列を返す"""

        # セッションステート上の初期値を用意
        if self.key_input not in st.session_state:
            st.session_state[self.key_input] = ""

        # テキストエリア本体
        st.write("")  # ちょっとだけ余白
        user_input = st.text_area(
            "あなたの発言を入力:",
            key=self.key_input,
            height=160,
        )

        # 送信ボタン
        send_clicked = st.button("送信", key=self.key_button)

        if send_clicked:
            text = st.session_state[self.key_input].strip()
            if text:
                # 入力を使い終わったらクリアする
                st.session_state[self.key_input] = ""
                return text

        return ""
