# components/player_input.py
import streamlit as st


class PlayerInput:
    """
    プレイヤーの発言入力欄と送信ボタンを管理するコンポーネント。

    ・ユーザー入力を安全に取得して返す
    ・送信ボタン押下後はテキストを自動的にクリア
    ・空文字は送信されないように防止
    """

    def __init__(
        self,
        key_input: str = "user_input_box",
        key_button: str = "send_button",
        label: str = "あなたの発言を入力:",
        height: int = 160,
    ):
        self.key_input = key_input
        self.key_button = key_button
        self.label = label
        self.height = height

    def render(self) -> str:
        """
        入力欄と送信ボタンを描画し、送信時のテキストを返す。
        戻り値が空文字列のときは送信なし。
        """
        # 入力欄を表示
        user_input = st.text_area(
            self.label,
            key=self.key_input,
            height=self.height,
        )

        # 送信ボタン
        send_clicked = st.button("送信", key=self.key_button)

        # ボタン押下時に入力を取得
        if send_clicked:
            text = user_input.strip()
            if text:
                # 入力内容を返す前にクリア
                st.session_state[self.key_input] = ""
                return text

        # 押されなかった or 空文字なら何も返さない
        return ""
