# views/council_view.py  （ファイル名が council1_view.py なら同じ中身でOK）
from __future__ import annotations

import streamlit as st

# ★ ここがポイント：actors. ではなく council. 配下から import する
from council.council_manager import (
    get_or_create_riseria_council_manager,
    # フローリア版も残したいなら↓も使える
    # get_or_create_floria_council_manager,
)


class CouncilView:
    """
    会談システムビュー。

    現状は「下級生エルフ：リセリア」との 1on1 (+ナレーション) 会話専用ビューとして構成。
    将来フローリア用に戻す場合は、get_or_create_floria_council_manager() を呼ぶ分岐を追加すればOK。
    """

    TITLE = "🗣 会談システム（β）"

    def __init__(self) -> None:
        pass

    def render(self) -> None:
            st.header(self.TITLE)

            # 将来プレイヤーネームを UI から変えたい場合は、
            # st.session_state などから拾う設計にしておく
            player_name = st.session_state.get("player_name", "アツシ")

            # ★ ここが一番大事：リセリア用 CouncilManager を取得
            council = get_or_create_riseria_council_manager(player_name=player_name)

            # そのまま CouncilManager に画面描画を委譲
            council.render()
