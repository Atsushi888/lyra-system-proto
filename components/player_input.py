# components/player_input.py

from typing import Optional
import streamlit as st


class PlayerInput:
    # テキストエリア用のキー
    TEXT_KEY = "player_input_text"
    # LyraEngine 側と合わせる
    SCROLL_FLAG_KEY = "scroll_to_input"

    def __init__(self) -> None:
        # 最初の一回だけ空文字で初期化（value= は使わない）
        if self.TEXT_KEY not in st.session_state:
            st.session_state[self.TEXT_KEY] = ""

    def render(self) -> str:
        """
        プレイヤー入力欄を表示し、「送信」されたテキストだけを返す。
        送信されなければ "" を返す。
        """

        # ここにアンカーを置いておくと、将来 JS でスクロールしやすい
        st.markdown("<div id='player-input-area'></div>", unsafe_allow_html=True)

        st.write("あなたの発言を入力:")

        # 🔸ここが重要：value= を渡さず、key だけで管理する
        user_text: str = st.text_area(
            label="",
            key=self.TEXT_KEY,
            height=160,
        )

        # ボタン行
        send = st.button("送信", type="primary")

        # 押されていて、かつ非空なら「今回の発言」として返す
        if send and user_text.strip():
            text_to_send = user_text

            # 次のターンのために入力欄をクリア
            # （key が既に存在しているのでこの代入はOK）
            st.session_state[self.TEXT_KEY] = ""

            # 次回レンダリング時に、LyraEngine 側で
            # scroll_to_input フラグを見てスクロールする想定
            # → フラグ自体のセットは LyraEngine.render() 側でやっている
            return text_to_send

        # 送信されなかった場合
        return ""
