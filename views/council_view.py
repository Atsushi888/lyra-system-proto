# views/council_view.py

from __future__ import annotations
from typing import Any, Dict

import streamlit as st

from council.council_manager import CouncilManager


class CouncilView:
    """
    会談システム（β）の画面側。
    CouncilManager のインスタンスは session_state に 1 個だけ持つ。
    """

    SESSION_KEY_MANAGER = "council_manager"

    def __init__(self) -> None:
        ...

    # --- Manager の取得 ---
    def _get_manager(self) -> CouncilManager:
        if self.SESSION_KEY_MANAGER not in st.session_state:
            st.session_state[self.SESSION_KEY_MANAGER] = CouncilManager()
        return st.session_state[self.SESSION_KEY_MANAGER]

    # --- 画面描画 ---
    def render(self) -> None:
        manager = self._get_manager()
        log = manager.get_log()
        state = manager.get_state()

        st.markdown("## 🗣️ 会談システム（Council Prototype）")
        st.caption("※ ロジックとUIは CouncilManager に集約。ここから拡張していく。")

        # 上部コントロール
        col_left, col_right = st.columns([3, 1])
        with col_right:
            if st.button("🔁 リセット", key="council_reset"):
                manager.reset()
                st.rerun()

        # 会談ログ
        st.markdown("### 会談ログ")
        if not log:
            st.caption("（まだ会談が始まっていません。プレイヤーが発言すると会談が始まります）")
        else:
            for idx, entry in enumerate(log, start=1):
                role = entry.get("role", "system")
                raw = entry.get("content", "")
                text = raw.replace("<br>", "\n")

                if role == "player":
                    name = "プレイヤー"
                elif role == "floria":
                    name = "フローリア"
                else:
                    name = role

                st.markdown(f"**[{idx}] {name}**")
                st.markdown(text)
                st.markdown("---")

        # サイドバーのステータス
        with st.sidebar.expander("📊 会談ステータス", expanded=True):
            st.write(f"ラウンド: {state.get('round')}")
            st.write(f"話者: {state.get('speaker')}")
            st.write(f"モード: {state.get('mode')}")
            st.write(f"参加者: { ' / '.join(state.get('participants', [])) }")
            last_sp = state.get("last_speaker") or "（なし）"
            st.write(f"最後の話者: {last_sp}")

        # プレイヤー入力
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
                if cleaned:
                    with st.spinner("フローリアは考えています…"):
                        manager.proceed(cleaned)
                    # 入力欄クリア
                    st.session_state["council_user_input"] = ""
                    st.rerun()
