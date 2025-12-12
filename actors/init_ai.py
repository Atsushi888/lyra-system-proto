# /mount/src/lyra-system-proto/actors/init_ai.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass
class InitAI:
    """
    セッション初期化の「唯一の正規窓口」。
    """

    DEFAULT_PLAYER_NAME = "アツシ"

    DEFAULT_DOKIPOWER_STATE: Dict[str, Any] = {
        "mode": "normal",
        "affection": 0.5,
        "arousal": 0.3,
        "doki_power": 0.0,
        "doki_level": 0,
        "relationship_level": 20,
        "masking_level": 30,
        "environment": "alone",
        "interaction_mode_hint": "auto",
    }

    DEFAULT_EMOTION_MANUAL_CONTROLS: Dict[str, Any] = {
        "relationship_level": 20,
        "doki_power": 0.0,
        "masking_level": 30,
        "environment": "alone",
        "others_present": False,
        "interaction_mode_hint": "auto",
    }

    DEFAULT_WORLD_STATE_MANUAL_CONTROLS: Dict[str, Any] = {
        "others_present": False,
        "interaction_mode_hint": "auto",
    }

    @classmethod
    def ensure_all(cls, *, state: Mapping[str, Any], persona: Optional[Any] = None) -> None:
        cls.ensure_player_name(state=state, persona=persona)
        cls.ensure_manual_controls(state=state)
        cls.ensure_world_state(state=state)

    @classmethod
    def ensure_player_name(cls, *, state: Mapping[str, Any], persona: Optional[Any] = None) -> str:
        player_name = state.get("player_name")
        if isinstance(player_name, str) and player_name.strip():
            return player_name.strip()

        if persona is not None:
            for key in ("player_name", "name", "player"):
                v = getattr(persona, key, None)
                if isinstance(v, str) and v.strip():
                    player_name = v.strip()
                    break

        if not (isinstance(player_name, str) and player_name.strip()):
            player_name = cls.DEFAULT_PLAYER_NAME

        state["player_name"] = player_name  # type: ignore[index]
        return player_name

    @classmethod
    def ensure_manual_controls(cls, *, state: Mapping[str, Any]) -> None:
        ds = state.get("dokipower_state")
        if not isinstance(ds, dict):
            ds = {}
        for k, v in cls.DEFAULT_DOKIPOWER_STATE.items():
            ds.setdefault(k, v)
        state["dokipower_state"] = ds  # type: ignore[index]

        emc = state.get("emotion_manual_controls")
        if not isinstance(emc, dict):
            emc = {}
        for k, v in cls.DEFAULT_EMOTION_MANUAL_CONTROLS.items():
            emc.setdefault(k, v)
        state["emotion_manual_controls"] = emc  # type: ignore[index]

        wmc = state.get("world_state_manual_controls")
        if not isinstance(wmc, dict):
            wmc = {}
        for k, v in cls.DEFAULT_WORLD_STATE_MANUAL_CONTROLS.items():
            wmc.setdefault(k, v)
        state["world_state_manual_controls"] = wmc  # type: ignore[index]

    @classmethod
    def ensure_world_state(cls, *, state: Mapping[str, Any]) -> None:
        player_name = cls.ensure_player_name(state=state, persona=None)

        ws = state.get("world_state")
        if not isinstance(ws, dict):
            ws = {}

        ws.setdefault("player_name", player_name)

        loc = ws.get("locations")
        if not isinstance(loc, dict):
            loc = {}

        default_room = f"{player_name}の部屋"
        loc.setdefault("player", default_room)
        loc.setdefault("floria", default_room)
        ws["locations"] = loc

        t = ws.get("time")
        if not isinstance(t, dict):
            t = {}
        t.setdefault("slot", "morning")
        t.setdefault("time_str", "07:30")
        ws["time"] = t

        others_present = ws.get("others_present")
        wmc = state.get("world_state_manual_controls")
        if isinstance(wmc, dict) and isinstance(wmc.get("others_present"), bool):
            others_present = wmc["others_present"]
        if not isinstance(others_present, bool):
            others_present = False
        ws["others_present"] = others_present

        ws.setdefault("weather", "clear")

        party = ws.get("party")
        if not isinstance(party, dict):
            party = {}
        party.setdefault(
            "mode",
            "both" if loc.get("player") == loc.get("floria") else "alone",
        )
        ws["party"] = party

        state["world_state"] = ws  # type: ignore[index]
