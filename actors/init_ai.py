from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass
class InitAI:
    """
    セッション初期化の「唯一の正規窓口」。

    目的:
    - player_name の単一ソース化
    - world_state の既定値初期化（player_name を反映）
    - manual_controls（emotion/world_state）の既定値初期化
    - DokiPowerController の dokipower_state も最低限の既定値を入れる
      （※UIを開かなくても下流が落ちないため）
    """

    DEFAULT_PLAYER_NAME = "アツシ"

    # dokipower_control.py と合わせる（キー名・意味を一致させる）
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
        """
        ここだけ呼べば OK の入口。
        """
        cls.ensure_player_name(state=state, persona=persona)
        cls.ensure_manual_controls(state=state)
        cls.ensure_world_state(state=state)

    # ------------------------------------------------------------
    # player_name
    # ------------------------------------------------------------
    @classmethod
    def ensure_player_name(cls, *, state: Mapping[str, Any], persona: Optional[Any] = None) -> str:
        # state は Mapping だが実体は st.session_state / dict を想定
        s = state  # alias

        # 1) すでに session にあるならそれを使う
        player_name = s.get("player_name")
        if isinstance(player_name, str) and player_name.strip():
            return player_name.strip()

        # 2) persona 側に player_name 的なものがあれば拾う（将来拡張用）
        #    ※今は無理に決め打ちせず、存在すれば使う
        if persona is not None:
            for key in ("player_name", "name", "player"):
                v = getattr(persona, key, None)
                if isinstance(v, str) and v.strip():
                    player_name = v.strip()
                    break

        # 3) 最後に既定値
        if not (isinstance(player_name, str) and player_name.strip()):
            player_name = cls.DEFAULT_PLAYER_NAME

        # 書き戻し
        try:
            state["player_name"] = player_name  # type: ignore[index]
        except Exception:
            pass

        return player_name

    # ------------------------------------------------------------
    # manual controls
    # ------------------------------------------------------------
    @classmethod
    def ensure_manual_controls(cls, *, state: Mapping[str, Any]) -> None:
        # dokipower_state（UIを開かなくても下流が参照しても安全にする）
        ds = state.get("dokipower_state")
        if not isinstance(ds, dict):
            ds = {}
        for k, v in cls.DEFAULT_DOKIPOWER_STATE.items():
            ds.setdefault(k, v)
        try:
            state["dokipower_state"] = ds  # type: ignore[index]
        except Exception:
            pass

        # emotion_manual_controls（← ここが今回の不足箇所）
        emc = state.get("emotion_manual_controls")
        if not isinstance(emc, dict):
            emc = {}
        for k, v in cls.DEFAULT_EMOTION_MANUAL_CONTROLS.items():
            emc.setdefault(k, v)
        try:
            state["emotion_manual_controls"] = emc  # type: ignore[index]
        except Exception:
            pass

        # world_state_manual_controls
        wmc = state.get("world_state_manual_controls")
        if not isinstance(wmc, dict):
            wmc = {}
        for k, v in cls.DEFAULT_WORLD_STATE_MANUAL_CONTROLS.items():
            wmc.setdefault(k, v)
        try:
            state["world_state_manual_controls"] = wmc  # type: ignore[index]
        except Exception:
            pass

    # ------------------------------------------------------------
    # world_state
    # ------------------------------------------------------------
    @classmethod
    def ensure_world_state(cls, *, state: Mapping[str, Any]) -> None:
        player_name = cls.ensure_player_name(state=state, persona=None)

        ws = state.get("world_state")
        if not isinstance(ws, dict):
            ws = {}

        # player_name を world_state にも保持（将来の一括参照用）
        ws.setdefault("player_name", player_name)

        # locations
        loc = ws.get("locations")
        if not isinstance(loc, dict):
            loc = {}

        default_room = f"{player_name}の部屋"
        loc.setdefault("player", default_room)
        loc.setdefault("floria", default_room)
        ws["locations"] = loc

        # time
        t = ws.get("time")
        if not isinstance(t, dict):
            t = {}
        t.setdefault("slot", "morning")
        t.setdefault("time_str", "07:30")
        ws["time"] = t

        # others_present（manual があれば最優先で反映）
        others_present = ws.get("others_present")
        wmc = state.get("world_state_manual_controls")
        if isinstance(wmc, dict) and isinstance(wmc.get("others_present"), bool):
            others_present = wmc["others_present"]
        if not isinstance(others_present, bool):
            others_present = False
        ws["others_present"] = others_present

        # weather
        ws.setdefault("weather", "clear")

        # party.mode
        party = ws.get("party")
        if not isinstance(party, dict):
            party = {}
        # player と floria が同じ場所なら both
        try:
            party_mode = "both" if loc.get("player") == loc.get("floria") else "alone"
        except Exception:
            party_mode = "both"
        party.setdefault("mode", party_mode)
        ws["party"] = party

        # 保存
        try:
            state["world_state"] = ws  # type: ignore[index]
        except Exception:
            pass
