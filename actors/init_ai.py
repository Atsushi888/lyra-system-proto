# actors/init_ai.py

from __future__ import annotations

from typing import Any, Dict, Mapping


class InitAI:
    """
    セッション初期化の一本化。

    - player_name / world_state
    - dokipower_state（UIスライダーの保持）
    - emotion_manual_controls / world_state_manual_controls
      （Mixer / Scene / Narrator が読む値）
    """

    # =========================
    # Session Keys
    # =========================
    KEY_PLAYER_NAME = "player_name"
    KEY_WORLD_STATE = "world_state"

    KEY_DOKI_STATE = "dokipower_state"
    KEY_EMO_MANUAL = "emotion_manual_controls"
    KEY_WS_MANUAL = "world_state_manual_controls"

    # =========================
    # Defaults
    # =========================
    DEFAULT_PLAYER_NAME = "アツシ"

    @staticmethod
    def default_dokipower_state() -> Dict[str, Any]:
        """
        components/dokipower_control.py の _get_state() と同等の既定値
        """
        return {
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

    # =========================================================
    # Public entry
    # =========================================================
    @classmethod
    def ensure_all(cls, *, state: Mapping[str, Any], persona: Any = None) -> None:
        """
        ここだけ呼べば初期化が完了する入口
        """
        cls.ensure_player_name(state=state)
        cls.ensure_dokipower_state(state=state)
        cls.ensure_manual_controls(state=state)
        cls.ensure_world_state(state=state, persona=persona)

    # =========================================================
    # 1) player_name
    # =========================================================
    @classmethod
    def ensure_player_name(cls, *, state: Mapping[str, Any]) -> str:
        st: Any = state
        name = st.get(cls.KEY_PLAYER_NAME)
        if not isinstance(name, str) or not name.strip():
            name = cls.DEFAULT_PLAYER_NAME
            st[cls.KEY_PLAYER_NAME] = name
        return name

    # =========================================================
    # 2) dokipower_state
    # =========================================================
    @classmethod
    def ensure_dokipower_state(cls, *, state: Mapping[str, Any]) -> Dict[str, Any]:
        st: Any = state
        ds = st.get(cls.KEY_DOKI_STATE)
        if not isinstance(ds, dict):
            ds = cls.default_dokipower_state()
            st[cls.KEY_DOKI_STATE] = ds
            return ds

        defaults = cls.default_dokipower_state()
        for k, v in defaults.items():
            ds.setdefault(k, v)

        st[cls.KEY_DOKI_STATE] = ds
        return ds

    # =========================================================
    # 3) emotion_manual_controls / world_state_manual_controls
    # =========================================================
    @classmethod
    def ensure_manual_controls(cls, *, state: Mapping[str, Any]) -> None:
        st: Any = state
        ds = cls.ensure_dokipower_state(state=state)

        relationship_level = int(ds.get("relationship_level", 20))
        doki_power = float(ds.get("doki_power", 0.0))
        masking_level = int(ds.get("masking_level", 30))

        environment = ds.get("environment", "alone")
        if environment not in ("alone", "with_others"):
            environment = "alone"

        interaction_mode_hint = ds.get("interaction_mode_hint", "auto")
        if interaction_mode_hint not in (
            "auto",
            "auto_with_others",
            "pair_private",
            "pair_public",
            "solo",
            "solo_with_others",
        ):
            interaction_mode_hint = "auto"

        others_present = (
            environment == "with_others"
            or interaction_mode_hint == "auto_with_others"
        )

        # ---- emotion_manual_controls
        em = st.get(cls.KEY_EMO_MANUAL)
        if not isinstance(em, dict):
            em = {}
        em.setdefault("relationship_level", relationship_level)
        em.setdefault("doki_power", doki_power)
        em.setdefault("masking_level", masking_level)
        em.setdefault("environment", environment)
        em.setdefault("others_present", others_present)
        em.setdefault("interaction_mode_hint", interaction_mode_hint)
        st[cls.KEY_EMO_MANUAL] = em

        # ---- world_state_manual_controls
        wm = st.get(cls.KEY_WS_MANUAL)
        if not isinstance(wm, dict):
            wm = {}
        wm.setdefault("others_present", others_present)
        wm.setdefault("interaction_mode_hint", interaction_mode_hint)
        st[cls.KEY_WS_MANUAL] = wm

    # =========================================================
    # 4) world_state
    # =========================================================
    @classmethod
    def ensure_world_state(cls, *, state: Mapping[str, Any], persona: Any = None) -> Dict[str, Any]:
        st: Any = state
        player_name = cls.ensure_player_name(state=state)
        cls.ensure_manual_controls(state=state)

        ws = st.get(cls.KEY_WORLD_STATE)
        if not isinstance(ws, dict):
            ws = {}

        ws.setdefault("player_name", player_name)

        # ---- locations
        loc = ws.get("locations")
        if not isinstance(loc, dict):
            loc = {}

        default_room = f"{player_name}の部屋"
        loc.setdefault("player", default_room)
        loc.setdefault("floria", default_room)  # 互換キー
        ws["locations"] = loc

        # ---- time
        t = ws.get("time")
        if not isinstance(t, dict):
            t = {}
        t.setdefault("slot", "morning")
        t.setdefault("time_str", "07:30")
        ws["time"] = t

        # ---- others_present
        others_present = ws.get("others_present")
        wm = st.get(cls.KEY_WS_MANUAL)
        if isinstance(wm, dict) and isinstance(wm.get("others_present"), bool):
            others_present = wm["others_present"]
        if not isinstance(others_present, bool):
            others_present = False
        ws["others_present"] = others_present

        # ---- weather
        ws.setdefault("weather", "clear")

        # ---- party
        party = ws.get("party")
        if not isinstance(party, dict):
            party = {}
        party.setdefault(
            "mode",
            "both" if loc.get("player") == loc.get("floria") else "alone",
        )
        ws["party"] = party

        st[cls.KEY_WORLD_STATE] = ws
        return ws
