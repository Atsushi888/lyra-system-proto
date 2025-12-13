# actors/init_ai.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


# ==========================================================
# InitAI: セッション初期化の唯一窓口
# - player_name / world_state / manual_controls / dokipower_state を一本化
# - AnswerTalker / SceneAI / NarratorAI などは「存在を前提」にできる
# ==========================================================


def _default_dokipower_state() -> Dict[str, Any]:
    # components/dokipower_control.py の初期値と揃える（依存は避けてコピー）
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


def _default_emotion_manual_controls() -> Dict[str, Any]:
    # 「適用ボタン未押下」でも Mixer/Scene/Narrator が読める最低限の形
    return {
        "relationship_level": 20,
        "doki_power": 0.0,
        "masking_level": 30,
        "environment": "alone",
        "others_present": False,
        "interaction_mode_hint": "auto",
    }


def _default_world_state_manual_controls() -> Dict[str, Any]:
    return {
        "others_present": False,
        "interaction_mode_hint": "auto",
    }


@dataclass
class InitSnapshot:
    """
    今後の拡張用：よく使う情報を構造体として受け取れる入口。
    （現状は InitAI.ensure_* が state を直接補完して動く）
    """
    player_name: str = "アツシ"
    partner_name: Optional[str] = None

    player_location: Optional[str] = None
    partner_location: Optional[str] = None

    time_slot: str = "morning"
    time_str: str = "07:30"
    others_present: bool = False
    weather: str = "clear"
    interaction_mode_hint: str = "auto"

    relationship_level: int = 20
    doki_power: float = 0.0
    masking_level: int = 30
    environment: str = "alone"


class InitAI:
    # dokipower_control.py と合わせる
    DOKIPOWER_SESSION_KEY = "dokipower_state"

    @classmethod
    def ensure_all(
        cls,
        *,
        state: Mapping[str, Any],
        persona: Any = None,
        snapshot: Optional[InitSnapshot] = None,
    ) -> None:
        """
        以前の呼び出し（InitAI.ensure_all）互換。
        """
        cls._ensure_player_name(state=state, persona=persona, snapshot=snapshot)
        cls._ensure_manual_controls(state=state, snapshot=snapshot)
        cls._ensure_world_state(state=state, snapshot=snapshot)

    @classmethod
    def ensure_minimum(
        cls,
        *,
        state: Mapping[str, Any],
        persona: Any = None,
        snapshot: Optional[InitSnapshot] = None,
    ) -> None:
        """
        AnswerTalker 側が InitAI.ensure_minimum を呼ぶ互換のための入口。
        「最低限」と言いつつ、現状は ensure_all と同義でOK。
        """
        cls.ensure_all(state=state, persona=persona, snapshot=snapshot)

    # ------------------------
    # player_name
    # ------------------------
    @classmethod
    def _ensure_player_name(
        cls,
        *,
        state: Mapping[str, Any],
        persona: Any = None,
        snapshot: Optional[InitSnapshot] = None,
    ) -> None:
        # state は Mapping だが実体は dict / SessionState を想定
        s = state  # alias

        # 1) snapshot 指定
        if snapshot and isinstance(snapshot.player_name, str) and snapshot.player_name.strip():
            player_name = snapshot.player_name.strip()
        else:
            # 2) 既存 session/state
            raw = getattr(s, "get", lambda k, d=None: d)("player_name", None)
            if isinstance(raw, str) and raw.strip():
                player_name = raw.strip()
            else:
                # 3) persona 側に player_name があれば拾う（無ければ規定値）
                p = getattr(persona, "player_name", None)
                player_name = str(p).strip() if isinstance(p, str) and str(p).strip() else "アツシ"

        # 保存
        try:
            s["player_name"] = player_name  # type: ignore[index]
        except Exception:
            # Mapping が本当に immutable なら何もしない
            pass

    # ------------------------
    # manual controls / dokipower
    # ------------------------
    @classmethod
    def _ensure_manual_controls(
        cls,
        *,
        state: Mapping[str, Any],
        snapshot: Optional[InitSnapshot] = None,
    ) -> None:
        s = state

        # dokipower_state（UIのスライダー元）
        dp = getattr(s, "get", lambda k, d=None: d)(cls.DOKIPOWER_SESSION_KEY, None)
        if not isinstance(dp, dict):
            dp = _default_dokipower_state()

            # snapshot があれば反映
            if snapshot:
                dp["relationship_level"] = int(snapshot.relationship_level)
                dp["doki_power"] = float(snapshot.doki_power)
                dp["masking_level"] = int(snapshot.masking_level)
                dp["environment"] = str(snapshot.environment)
                dp["interaction_mode_hint"] = str(snapshot.interaction_mode_hint)

            try:
                s[cls.DOKIPOWER_SESSION_KEY] = dp  # type: ignore[index]
            except Exception:
                pass

        # emotion_manual_controls（Mixer/Scene/Narrator が読む）
        emc = getattr(s, "get", lambda k, d=None: d)("emotion_manual_controls", None)
        if not isinstance(emc, dict):
            emc = _default_emotion_manual_controls()

            # dokipower_state があるならそこから同期（未適用でも最低限の整合）
            if isinstance(dp, dict):
                emc["relationship_level"] = int(dp.get("relationship_level", emc["relationship_level"]))
                emc["doki_power"] = float(dp.get("doki_power", emc["doki_power"]))
                emc["masking_level"] = int(dp.get("masking_level", emc["masking_level"]))
                emc["environment"] = str(dp.get("environment", emc["environment"]))
                emc["interaction_mode_hint"] = str(dp.get("interaction_mode_hint", emc["interaction_mode_hint"]))

            # snapshot 優先
            if snapshot:
                emc["relationship_level"] = int(snapshot.relationship_level)
                emc["doki_power"] = float(snapshot.doki_power)
                emc["masking_level"] = int(snapshot.masking_level)
                emc["environment"] = str(snapshot.environment)
                emc["interaction_mode_hint"] = str(snapshot.interaction_mode_hint)
                emc["others_present"] = bool(snapshot.others_present)

            # others_present 推定（environment と矛盾しないように）
            if not isinstance(emc.get("others_present"), bool):
                env = emc.get("environment")
                if env == "with_others":
                    emc["others_present"] = True
                else:
                    emc["others_present"] = False

            try:
                s["emotion_manual_controls"] = emc  # type: ignore[index]
            except Exception:
                pass
        else:
            # 既に dict がある場合でも、欠けてるキーだけ補完（今回のバグ対策）
            emc.setdefault("relationship_level", 20)
            emc.setdefault("doki_power", 0.0)
            emc.setdefault("masking_level", 30)
            emc.setdefault("environment", "alone")
            emc.setdefault("interaction_mode_hint", "auto")
            emc.setdefault("others_present", False)

        # world_state_manual_controls（world_state 用フラグ）
        wsm = getattr(s, "get", lambda k, d=None: d)("world_state_manual_controls", None)
        if not isinstance(wsm, dict):
            wsm = _default_world_state_manual_controls()

            # emotion_manual_controls から最低限同期
            if isinstance(emc, dict):
                if isinstance(emc.get("others_present"), bool):
                    wsm["others_present"] = emc["others_present"]
                hint = emc.get("interaction_mode_hint")
                if isinstance(hint, str):
                    wsm["interaction_mode_hint"] = hint

            # snapshot 優先
            if snapshot:
                wsm["others_present"] = bool(snapshot.others_present)
                wsm["interaction_mode_hint"] = str(snapshot.interaction_mode_hint)

            try:
                s["world_state_manual_controls"] = wsm  # type: ignore[index]
            except Exception:
                pass
        else:
            wsm.setdefault("others_present", False)
            wsm.setdefault("interaction_mode_hint", "auto")

    # ------------------------
    # world_state
    # ------------------------
    @classmethod
    def _ensure_world_state(
        cls,
        *,
        state: Mapping[str, Any],
        snapshot: Optional[InitSnapshot] = None,
    ) -> None:
        s = state

        player_name = getattr(s, "get", lambda k, d=None: d)("player_name", "アツシ")
        if not isinstance(player_name, str) or not player_name.strip():
            player_name = "アツシ"
        player_name = player_name.strip()

        ws = getattr(s, "get", lambda k, d=None: d)("world_state", None)
        if not isinstance(ws, dict):
            ws = {}

        # player_name を world_state にも保持（デバッグ・プロンプト用）
        ws.setdefault("player_name", player_name)

        # locations
        loc = ws.get("locations") or {}
        if not isinstance(loc, dict):
            loc = {}

        default_room = f"{player_name}の部屋"

        player_loc = None
        partner_loc = None
        if snapshot:
            player_loc = snapshot.player_location
            partner_loc = snapshot.partner_location

        loc["player"] = (player_loc or loc.get("player") or default_room)
        loc["floria"] = (partner_loc or loc.get("floria") or default_room)  # 互換：floriaキーは残す
        ws["locations"] = loc

        # time
        t = ws.get("time") or {}
        if not isinstance(t, dict):
            t = {}
        if snapshot:
            t["slot"] = snapshot.time_slot or t.get("slot") or "morning"
            t["time_str"] = snapshot.time_str or t.get("time_str") or "07:30"
        else:
            t.setdefault("slot", "morning")
            t.setdefault("time_str", "07:30")
        ws["time"] = t

        # others_present は manual_controls を優先
        wsm = getattr(s, "get", lambda k, d=None: d)("world_state_manual_controls", None)
        if isinstance(wsm, dict) and isinstance(wsm.get("others_present"), bool):
            ws["others_present"] = wsm["others_present"]
        else:
            # snapshot があれば採用、なければ既存→False
            if snapshot:
                ws["others_present"] = bool(snapshot.others_present)
            else:
                v = ws.get("others_present")
                ws["others_present"] = bool(v) if isinstance(v, bool) else False

        # weather
        if snapshot and isinstance(snapshot.weather, str) and snapshot.weather:
            ws["weather"] = snapshot.weather
        else:
            ws.setdefault("weather", "clear")

        # party.mode
        party = ws.get("party") or {}
        if not isinstance(party, dict):
            party = {}
        party["mode"] = "both" if loc.get("player") == loc.get("floria") else "alone"
        ws["party"] = party

        try:
            s["world_state"] = ws  # type: ignore[index]
        except Exception:
            pass
