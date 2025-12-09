from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # 型ヒント専用（実行時には import しない）
    from .narrator_ai import NarratorAI, NarrationChoice


def make_special_title_and_choice_impl(
    narrator: "NarratorAI",
    special_id: str,
    world_state: Dict[str, Any] | None = None,
    floria_state: Dict[str, Any] | None = None,
) -> tuple[str, "NarrationChoice"]:
    """
    スペシャルアクション用タイトルと Choice を構築する共通実装。

    world_state は呼び出し側で SceneAI によって管理される想定。
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

    from .narrator_ai import NarrationChoice

    choice = NarrationChoice(
        kind="special",
        label=title,
        speak_text=speak,
        special_id=special_id,
    )
    return title, choice
