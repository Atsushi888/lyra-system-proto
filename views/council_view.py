# views/council_view.py

from __future__ import annotations
from typing import Any

import streamlit as st

from council.council_manager import CouncilManager


class CouncilView:
    """
    会談システム画面（β）。
    CouncilManager を session_state に 1 つ持って、UI だけ担当する。
    """

    SESSION_MANAGER = "council_manager"

    # ---- manager 取得 ----
    def _get_manager(self) -> CouncilManager:
        if self.SESSION_MANAGER not in st.session_state:
            st.session_state[self.SESSION_MANAGER] = CouncilManager()
        return st.session_state[self.SESSION_MANAGER]

    # ---- 画面描画 ----
    def render(self) -> None:
        manager = self._get_manager()
        log = manager.get_log()
        status = manager.get_status()

        st.markdown("## 🗣️ 会談システム（Council Prototype）")
        st.caption("※ Actor ベースで AI と会話する会談システム（β）です。")

        # 上部コントロール
        col_left, col_right = st.columns([3, 1])
        with col_right:
            if st.button("🔁 リセット", key="council_reset"):
                manager.reset()
                st.success("会談をリセットしました。")
                st.rerun()

        # ---- 会談ログ ----
        st.markdown("### 会談ログ")
        if not log:
            st.caption("（まだ会談は始まっていません。何か話しかけてみましょう）")
        else:
            for idx, entry in enumerate(log, start=1):
                role = entry.get("role", "")
                text = entry.get("content", "")
                if role == "player":
                    name = "プレイヤー"
                elif role == "floria":
                    name = "フローリア"
                else:
                    name = role or "？"

                st.markdown(f"**[{idx}] {name}**")
                # <br> を有効にするため unsafe_allow_html=True
                st.markdown(text, unsafe_allow_html=True)
                st.markdown("---")

        # ---- サイドバー：会談ステータス ----
        with st.sidebar.expander("📊 会談ステータス", expanded=True):
            st.write(f"ラウンド: {status.get('round')}")
            st.write(f"話者: {status.get('speaker')}")
            st.write(f"モード: {status.get('mode')}")
            participants = status.get("participants") or []
            if participants:
                st.write("参加者: " + " / ".join(participants))
            last = status.get("last_speaker")
            if last:
                st.write(f"最後の話者: {last}")

        # ---- プレイヤー入力 ----
        st.markdown("### プレイヤー入力")

        user_text = st.text_area(
            "あなたの発言：",
            key="council_user_input",
            placeholder="ここにフローリアへの発言を書いてください。",
        )

        send_col, _ = st.columns([1, 3])
        with send_col:
            if st.button("送信", key="council_send"):
                cleaned = (user_text or "").strip()
                if not cleaned:
                    st.warning("発言を入力してください。")
                else:
                    # ★ フローリア思考中スピナー
                    with st.spinner("フローリアは少し考えています…"):
                        manager.proceed(cleaned)

                    # 入力欄クリア
                    st.session_state["council_user_input"] = ""
                    st.rerun()
