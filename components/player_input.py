# components/player_input.py
import streamlit as st


class PlayerInput:
    """ユーザーの入力欄＋送信処理を担当"""

    def __init__(
        self,
        key_input: str = "user_input_box",
        key_button: str = "send_btn",
        key_submitted: str = "user_input_submitted",
    ):
        self.key_input = key_input
        self.key_button = key_button
        self.key_submitted = key_submitted

    def render(self) -> str:
        """入力欄を描画して、送信ボタンが押されたときだけ文字列を返す"""

        # セッションステートの初期化
        if self.key_input not in st.session_state:
            st.session_state[self.key_input] = ""
        if self.key_submitted not in st.session_state:
            st.session_state[self.key_submitted] = ""

        # --- コールバック関数（ボタンが押されたときにだけ呼ばれる） ---
        def _on_click():
            # 入力済みテキストを「submitted」に退避してからクリア
            st.session_state[self.key_submitted] = st.session_state[self.key_input]
            st.session_state[self.key_input] = ""

        # テキストエリア本体
        st.text_area(
            "あなたの発言を入力:",
            key=self.key_input,
            height=160,
        )

        # 送信ボタン（押されたときに _on_click が実行される）
        st.button("送信", key=self.key_button, on_click=_on_click)

        # コールバックが退避させたテキストを取り出す
        text = st.session_state.get(self.key_submitted, "")
        # 一度返したらすぐ消す（「使い切りキュー」にする）
        st.session_state[self.key_submitted] = ""
        return text.strip()
