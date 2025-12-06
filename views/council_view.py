# views/council_view.py
from __future__ import annotations

import os
import streamlit as st

# ★ ここを "actors.council_manager" に統一する
from actors.council_manager import (
    get_or_create_riseria_council_manager,
    # フローリア版も残したいなら↓も使える
    # get_or_create_floria_council_manager,
)

LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"


class CouncilView:
    """
    会談システムビュー。

    現状は「下級生エルフ：リセリア」との 1on1 (+ナレーション) 会話専用ビューとして構成。
    将来フローリア用に戻す場合は、get_or_create_floria_council_manager() を呼ぶ分岐を追加すればOK。
    """

    TITLE = "🗣 会談システム（β）"

    def __init__(self) -> None:
        if LYRA_DEBUG:
            st.caption("[DEBUG:CouncilView] init CouncilView()")

    def render(self) -> None:
        st.header(self.TITLE)

        # 将来プレイヤーネームを UI から変えたい場合は、
        # st.session_state などから拾う設計にしておく
        player_name = st.session_state.get("player_name", "アツシ")

        if LYRA_DEBUG:
            st.caption(f"[DEBUG:CouncilView] player_name={player_name}")

        # ★ ここが一番大事：リセリア用 CouncilManager を取得
        council = get_or_create_riseria_council_manager(player_name=player_name)

        if LYRA_DEBUG:
            try:
                log_len = len(council.get_log())
            except Exception:
                log_len = "?"
            st.caption(
                f"[DEBUG:CouncilView] use CouncilManager(id={id(council)}), "
                f"log_len={log_len}"
            )

        # そのまま CouncilManager に画面描画を委譲
        council.render()
