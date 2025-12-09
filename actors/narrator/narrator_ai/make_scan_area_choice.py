from __future__ import annotations

from typing import Dict, Any, Optional, TYPE_CHECKING

from .narrator_ai import NarrationChoice

if TYPE_CHECKING:
    from .narrator_ai import NarratorAI


def build_scan_area_choice(
    narrator: "NarratorAI",
    location_name: str = "この場",
    world_state: Dict[str, Any] | None = None,
    floria_state: Dict[str, Any] | None = None,
) -> NarrationChoice:
    """
    「周囲の様子を見る」。
    party_mode==alone かつ外野なしなら「静かな情景」寄りにする。
    """
    snap = narrator._get_scene_snapshot()
    party_mode = snap["party_mode"]
    loc = snap["player_location"] or location_name
    others_present: bool = bool(snap.get("others_present", False))

    if party_mode == "alone" and not others_present:
        intent = f"{loc}の周囲の静かな様子を改めて見回す"
    else:
        intent = f"{loc}の周囲の様子を見回す"

    speak = narrator._refine(intent_text=intent, label="scan_area")

    return NarrationChoice(
        kind="scan_area",
        label="周囲の様子を見る",
        speak_text=speak,
        meta={"location": loc},
    )


def build_look_person_choice(
    narrator: "NarratorAI",
    actor_name: str = "フローリア",
    actor_id: Optional[str] = None,
    world_state: Dict[str, Any] | None = None,
    floria_state: Dict[str, Any] | None = None,
) -> NarrationChoice:
    """
    「相手の様子を伺う」。
    プレイヤー一人（party_mode == "alone"）のときは
    そもそも相手がいない旨を返す。
    """
    snap = narrator._get_scene_snapshot()
    party_mode = snap["party_mode"]

    # 互換のためデフォルト値は "フローリア" だが、
    # 別キャラがバインドされている場合はそちらを優先。
    if actor_name == "フローリア" and narrator.partner_name != "フローリア":
        actor_name = narrator.partner_name

    if party_mode == "alone":
        speak = (
            "周囲を見回してみるが、この場にあなた以外の人影はない。"
            "様子をうかがう相手が現れるのを、もう少し待つしかなさそうだ。"
        )
    else:
        intent = f"{actor_name}の様子をうかがう"
        speak = narrator._refine(intent_text=intent, label="look_person")

    return NarrationChoice(
        kind="look_person",
        label=f"{actor_name}の様子をうかがう",
        speak_text=speak,
        target_id=actor_id,
        meta={"actor_name": actor_name},
    )
