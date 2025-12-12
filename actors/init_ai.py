# actors/init_ai.py
from __future__ import annotations

from typing import Mapping, Any
import streamlit as st

from actors.scene.world_context import WorldContext


class InitAI:
    """
    WorldContext を唯一の入力として、
    session_state / world_state / manual_controls を初期化する。
    """

    @staticmethod
    def apply(context: WorldContext, state: Mapping[str, Any] | None = None) -> None:
        if state is None:
            state = st.session_state

        # =========================
        # world_state 本体
        # =========================
        state["world_state"] = {
            "player_name": context.player_name,
            "locations": {
                "player": context.player_location,
                "floria": context.partner_location,
            },
            "time": {
                "slot": context.time_slot,
                "time_str": context.time_str,
            },
            "others_present": context.others_present,
            "weather": context.weather,
            "party": {
                "mode": context.party_mode,
            },
        }

        # =========================
        # world_state_manual_controls
        # =========================
        state["world_state_manual_controls"] = {
            "others_present": context.others_present,
            "interaction_mode_hint": "auto",
        }

        # =========================
        # emotion_manual_controls（最低限）
        # =========================
        state.setdefault("emotion_manual_controls", {
            "environment": "with_others" if context.others_present else "alone",
            "others_present": context.others_present,
            "interaction_mode_hint": "auto",
        })

        # =========================
        # 安全弁：Round0 用
        # =========================
        state.setdefault("round_number", 0)
