# actors/narrator/narrator_ai/make_special_title_and_choice.py
from __future__ import annotations

from typing import Dict, Any, TYPE_CHECKING, Tuple

from .types import NarrationChoice

if TYPE_CHECKING:
    from .narrator_ai import NarratorAI


def make_special_title_and_choice_impl(
    self: "NarratorAI",
    special_id: str,
    world_state: Dict[str, Any] | None = None,
    floria_state: Dict[str, Any] | None = None,
) -> Tuple[str, NarrationChoice]:
    """
    スペシャルアクションは、プレイヤー単独でも実行可能。
    world_state は SceneAI から取得したものを前提に Refine する。
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

    speak = self._refine(intent_text=intent, label=f"special:{special_id}")  # type: ignore[attr-defined]

    choice = NarrationChoice(
        kind="special",
        label=title,
        speak_text=speak,
        special_id=special_id,
    )
    return title, choice
