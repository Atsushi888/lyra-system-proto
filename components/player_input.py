# components/player_input.py

import streamlit as st


class PlayerInput:
    TEXT_KEY = "player_input_text"
    CLEAR_FLAG_KEY = "player_input_clear_flag"

    def render(self) -> str:
        # ① 「クリアフラグ」が立っていたら、ウィジェットを作る前に state を消す
        if st.session_state.get(self.CLEAR_FLAG_KEY, False):
            st.session_state.pop(self.TEXT_KEY, None)
            st.session_state[self.CLEAR_FLAG_KEY] = False

        # ② 入力欄本体
        user_text: str = st.text_area(
            "あなたの発言を入力：",
            key=self.TEXT_KEY,
            height=160,
        )

        send = st.button("送信", type="primary")

        if send:
            text_to_send = (user_text or "").strip()
            if not text_to_send:
                return ""

            # ③ 次の rerun で入力欄を空にするためのフラグだけ立てておく
            st.session_state[self.CLEAR_FLAG_KEY] = True
            return text_to_send

        return ""
