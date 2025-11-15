# views/council_view.py

from __future__ import annotations
from typing import Any

import streamlit as st

from council.council_manager import CouncilManager


class CouncilView:
    """
    会談システムの画面側。
    ロジックは CouncilManager に任せて、ここでは
    - ログ表示
    - 入力欄
    - ステータス表示
    だけを担当する。
    """

    def __init__(self) -> None:
        self.manager = CouncilManager()

    def render(self) -> None:
        mgr = self.manager
        state = mgr.state
        log = mgr.conversation_log

        # ===== ヘッダ & リセットボタン =====
        col_title, col_btn = st.columns([3, 1])
        with col_title:
            st.markdown("## 🗣️ 会談システム（Council Prototype）")
        with col_btn:
            if st.button("🔁 リセット", key="council_reset"):
                mgr.reset()
                st.rerun()

        # ===== 会談ログ =====
        st.markdown("### 会談ログ")
        if not log:
            st.caption("（まだ発言がありません。メッセージを送ってみてください）")
        else:
            for i, entry in enumerate(log, start=1):
                role = entry.get("role", "?")
                content = entry.get("content", "")

                if role == "player":
                    name = "プレイヤー"
                elif role == "floria":
                    name = "フローリア"
                else:
                    name = role

                st.markdown(f"**[{i}] {name}**")
                # "  \n" を含む Markdown テキストとして描画
                st.markdown(content)
                st.markdown("---")

        # ===== サイドバー：ステータス =====
        with st.sidebar.expander("会談ステータス", expanded=True):
            st.write(f"ラウンド: {state.get('round', 1)}")
            st.write(f"話者: {state.get('speaker', 'player')}")
            st.write(f"モード: {state.get('mode', 'ongoing')}")
            # 参加者一覧を明示
            st.write("参加者: プレイヤー / フローリア")
            last_speaker = state.get("last_speaker")
            if last_speaker:
                st.write(f"最後の話者: {last_speaker}")

        # ===== プレイヤー入力 =====
        st.markdown("### プレイヤー入力")

        if state.get("mode") != "ongoing":
            st.caption("（会談は停止中です。「リセット」で再開してください）")
            return

        user_key = "council_user_input"
        user_text: str = st.text_area(
            "あなたの発言：",
            key=user_key,
            placeholder="ここにフローリアへの発言を書いてください。",
        )

        if st.button("送信", key="council_send"):
            text = (st.session_state.get(user_key) or "").strip()
            if text:
                mgr.proceed(text)
                # 入力欄をクリア
                st.session_state[user_key] = ""
                st.rerun()
