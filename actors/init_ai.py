# actors/init_ai.py
from __future__ import annotations
from typing import Any, Mapping

import streamlit as st


class InitAI:
    """
    セッション初期化の唯一の正規窓口。
    AnswerTalker / SceneAI / NarratorAI より前に一度だけ呼ばれる想定。
    """

    @classmethod
    def ensure_all(
        cls,
        *,
        state: Mapping[str, Any],
        persona: Any | None = None,
    ) -> None:
        """
        Lyra セッションに必要な初期状態をすべて保証する。
        """
        cls.ensure_player_name(state)
        cls.ensure_world_state(state, persona)
        cls.ensure_manual_controls(state)

    # -------------------------
    # 個別 ensure 群
    # -------------------------

    @staticmethod
    def ensure_player_name(state: Mapping[str, Any]) -> None:
        if "player_name" not in state:
            state["player_name"] = "アツシ"

    @staticmethod
    def ensure_world_state(
        state: Mapping[str, Any],
        persona: Any | None,
    ) -> None:
        ws = state.get("world_state")
        if not isinstance(ws, dict):
            player_name = state.get("player_name", "プレイヤー")

            ws = {
                "player_name": player_name,
                "locations": {
                    "player": f"{player_name}の部屋",
                    "floria": f"{player_name}の部屋",
                },
                "time": {
                    "slot": "morning",
                    "time_str": "07:30",
                },
                "others_present": False,
                "weather": "clear",
                "party": {
                    "mode": "both",
                },
            }
            state["world_state"] = ws

    @staticmethod
    def ensure_manual_controls(state: Mapping[str, Any]) -> None:
        state.setdefault(
            "world_state_manual_controls",
            {
                "others_present": False,
                "interaction_mode_hint": "auto",
            },
        )

        state.setdefault(
            "emotion_manual_controls",
            {},
        )
