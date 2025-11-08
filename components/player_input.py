# components/player_input.py
from __future__ import annotations

from dataclasses import dataclass
import streamlit as st


@dataclass
class PlayerInput:
    """
    プレイヤー入力欄。
    - テキストエリア（multi-line）
    - 「送信」ボタン
    - 送信後は次ターンでテキストをクリア
    """

    TEXT_KEY: str = "player_input_text"
    CLEAR_FLAG_KEY: str = "player_input_clear_next"

    def clear_next_turn(self) -> None:
        """次の rerun で入力欄をクリアさせるためのフラグを立てる。"""
        st.session_state[self.CLEAR_FLAG_KEY] = True

    def render(self) -> str:
        # ① もし「次ターンでクリア」フラグが立っていたら、先に消す
        if st.session_state.get(self.CLEAR_FLAG_KEY, False):
            st.session_state.pop(self.TEXT_KEY, None)
            st.session_state[self.CLEAR_FLAG_KEY] = False

        # ラベルだけ別に出して、余白を詰める
        st.markdown("**あなたの発言を入力：**")

        # ② 入力欄本体
        user_text: str = st.text_area(
            "",
            key=self.TEXT_KEY,
            height=160,
            label_visibility="collapsed",
        )

        # ③ 送信ボタン
        submitted = st.button("送信", type="primary")

        if submitted:
            # 前後の空白だけ削る
            return user_text.strip()

        return ""
