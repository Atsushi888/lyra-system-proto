from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .narrator_ai import NarratorAI


def make_scan_area_choice_impl(
    narrator: "NarratorAI",
    *,
    location_name: str = "この場",
    world_state: Dict[str, Any] | None = None,
    floria_state: Dict[str, Any] | None = None,
) -> Tuple[str, str]:
    """
    NarratorAI.make_scan_area_choice() の実装本体。
    戻り値: (speak_text, 実際に使ったロケーション名)
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
    return speak, loc
