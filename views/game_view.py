# views/game_view.py
from __future__ import annotations

import streamlit as st
from lyra_engine import LyraEngine


class GameView:
    def __init__(self) -> None:
        self.engine = LyraEngine()

    def render(self) -> None:
        # UserSettings 由来の情報を軽く表示（無ければデフォルト）
        settings = st.session_state.get("user_settings") or {}
        player_name = settings.get(
            "player_name", st.session_state.get("player_name", "アツシ")
        )
        length_mode = settings.get(
            "reply_length_mode", st.session_state.get("reply_length_mode", "auto")
        )

        mode_label_map = {
            "auto": "auto",
            "short": "short（短め）",
            "normal": "normal（ふつう）",
            "long": "long（少し長め）",
            "story": "story（ミニシーン）",
        }
        lm_label = mode_label_map.get(length_mode, length_mode)

        st.caption(f"プレイヤー名: {player_name} / 発話長さモード: {lm_label}")

        # メインのゲーム画面描画
        self.engine.render()
