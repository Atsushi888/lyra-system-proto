# components/player_input.py
from dataclasses import dataclass
import streamlit as st


@dataclass
class PlayerInput:
    TEXT_KEY: str = "player_input_text"
    CLEAR_FLAG_KEY: str = "player_input_clear_next"

    def render(self) -> str:
        """プレイヤーの入力欄と送信ボタン。
        送信後の rerun でテキストをクリアする。
        """

        # --- ① クリアフラグが立っていたら、ウィジェット生成 前 に消す ---
        if st.session_state.get(self.CLEAR_FLAG_KEY, False):
            st.session_state[self.TEXT_KEY] = ""
            st.session_state[self.CLEAR_FLAG_KEY] = False

        # --- ② ラベル + 入力欄 ---
        st.markdown("**あなたの発言を入力：**")

        user_text: str = st.text_area(
            "",
            key=self.TEXT_KEY,
            height=160,
            label_visibility="collapsed",
        )

        # --- ③ 送信ボタン ---
        submitted = st.button("送信", type="primary")

        if submitted:
            text_to_return = user_text.strip()

            # 次の run でクリアさせるフラグを立てる
            st.session_state[self.CLEAR_FLAG_KEY] = True

            return text_to_return

        return ""
