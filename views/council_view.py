# views/council_view.py

from __future__ import annotations
from typing import cast

import streamlit as st

from council.council_manager import CouncilManager


class CouncilView:
    """
    会談システム（Council Prototype）の UI レイヤ。

    - CouncilManager インスタンスを st.session_state で保持し、
      ログと状態を読みながら画面を構成する。
    """

    SESSION_KEY_MANAGER = "council_manager"
    SESSION_KEY_INPUT = "council_user_input"

    def __init__(self) -> None:
        # ここでは特に状態は持たず、render 時に session_state から Manager を取得する
        pass

    # ---- 内部ヘルパ ----

    def _get_manager(self) -> CouncilManager:
        if self.SESSION_KEY_MANAGER not in st.session_state:
            st.session_state[self.SESSION_KEY_MANAGER] = CouncilManager()
        return cast(CouncilManager, st.session_state[self.SESSION_KEY_MANAGER])

    # ---- 画面描画 ----

    def render(self) -> None:
        manager = self._get_manager()

        # ===== ヘッダ =====
        st.markdown("## 🗣️ 会談システム（β）")
        st.markdown("### 🗣️ 会談システム（Council Prototype）")
        st.caption("※ ロジックと UI は CouncilManager に集約。ここから拡張していく。")

        # ===== 上部コントロール（リセット） =====
        col_left, col_right = st.columns([3, 1])
        with col_right:
            if st.button("🔁 リセット", key="council_reset"):
                manager.reset()
                # 入力欄もクリア
                if self.SESSION_KEY_INPUT in st.session_state:
                    st.session_state[self.SESSION_KEY_INPUT] = ""
                st.experimental_rerun()

        # ===== 会談ログ =====
        st.markdown("### 会談ログ")

        if not manager.conversation_log:
            st.caption("（まだ発言がありません。プレイヤーとして話しかけてみてね）")
        else:
            for idx, entry in enumerate(manager.conversation_log, start=1):
                role = entry.get("role", "system")
                text = entry.get("content", "")

                if role == "player":
                    name = "プレイヤー"
                elif role == "floria":
                    name = "フローリア"
                else:
                    name = "システム"

                st.markdown(f"**[{idx}] {name}**")
                # <br> をそのまま改行として扱いたいので unsafe_allow_html=True
                st.markdown(text, unsafe_allow_html=True)
                st.markdown("---")

        # ===== サイドバー：会談ステータス =====
        with st.sidebar.expander("🧾 会談ステータス", expanded=True):
            st.write(f"ラウンド: {manager.state.get('round', 0)}")
            st.write(f"話者: {manager.state.get('speaker', '-')}")
            st.write(f"モード: {manager.state.get('mode', '-')}")
            participants = manager.state.get("participants") or []
            if participants:
                st.write("参加者: " + "／".join(participants))
            last_speaker = manager.state.get("last_speaker") or "（なし）"
            st.write(f"最後の話者: {last_speaker}")

        # ===== プレイヤー入力 =====
        st.markdown("### プレイヤー入力")

        if manager.state.get("mode") != "ongoing":
            st.caption("（現在この会談は終了状態です。リセットしてやり直してね）")
            return

        if manager.state.get("speaker") != "player":
            st.caption("（いまはプレイヤーのターンではありません）")
            return

        user_text = st.text_area(
            "あなたの発言：",
            key=self.SESSION_KEY_INPUT,
            placeholder="ここにフローリアへの発言を書いてください。",
        )

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            if st.button("送信", key="council_send"):
                text = (user_text or "").strip()
                if text:
                    # ★ ここで「フローリアは考えています…」メッセージを出す
                    with st.spinner("フローリアは考えています…"):
                        manager.proceed(text)

                    # 入力欄クリア
                    st.session_state[self.SESSION_KEY_INPUT] = ""

                st.experimental_rerun()
