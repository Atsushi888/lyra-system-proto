# actors/narrator/narrator_ai/make_scan_area_choice.py
from __future__ import annotations

from typing import Dict, Any

from .narrator_ai import NarratorAI, NarrationChoice


def make_scan_area_choice_impl(
    narrator: NarratorAI,
    location_name: str = "この場",
    world_state: Dict[str, Any] | None = None,
    floria_state: Dict[str, Any] | None = None,
) -> NarrationChoice:
    """
    「周囲の様子を見る」。
    プレイヤー一人でも有効だが、
    party_mode==alone の場合は「誰もいない静かな情景」を中心に描く。
    """
    snap = narrator._get_scene_snapshot()
    party_mode = snap["party_mode"]
    loc = snap["player_location"] or location_name

    if party_mode == "alone":
        # 一人のときは、静かなロケーション描写に寄せる
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
