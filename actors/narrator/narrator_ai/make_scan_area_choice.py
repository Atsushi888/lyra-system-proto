from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # 型ヒント専用（実行時には import しない）
    from .narrator_ai import NarratorAI, NarrationChoice


def make_scan_area_choice_impl(
    narrator: "NarratorAI",
    location_name: str = "この場",
    world_state: Dict[str, Any] | None = None,
    floria_state: Dict[str, Any] | None = None,
) -> "NarrationChoice":
    """
    「周囲の様子を見る」救済アクションの共通実装。

    - party_mode == "alone":
        → 静かなロケーション描写寄り（誰もいない空気感）
    - それ以外:
        → 通常の「周囲の様子を見回す」地の文
    """
    snap = narrator._get_scene_snapshot()
    party_mode = snap["party_mode"]
    loc = snap.get("player_location") or location_name

    if party_mode == "alone":
        intent = f"{loc}の周囲の静かな様子を改めて見回す"
    else:
        intent = f"{loc}の周囲の様子を見回す"

    speak = narrator._refine(intent_text=intent, label="scan_area")

    from .narrator_ai import NarrationChoice

    return NarrationChoice(
        kind="scan_area",
        label="周囲の様子を見る",
        speak_text=speak,
        meta={"location": loc},
    )
