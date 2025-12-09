from __future__ import annotations

from typing import Dict, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .narrator_ai import NarratorAI


def make_special_title_and_choice_impl(
    narrator: "NarratorAI",
    *,
    special_id: str,
    world_state: Dict[str, Any] | None = None,
    floria_state: Dict[str, Any] | None = None,
) -> Tuple[str, str]:
    """
    NarratorAI.make_special_title_and_choice() の実装本体。
    戻り値: (title, speak_text)
    """
    if special_id == "touch_pillar":
        title = "古い石柱に手を触れる"
        intent = "目の前の古い石柱に手を伸ばして触れる"
    elif special_id == "pray_to_moon":
        title = "月へ祈りを捧げる"
        intent = "静かに目を閉じて月へ祈りを捧げる"
    else:
        title = "特別な行動を取る"
        intent = "胸の内の衝動に従い、特別な行動をひとつ取る"

    speak = narrator._refine(intent_text=intent, label=f"special:{special_id}")
    return title, speak
