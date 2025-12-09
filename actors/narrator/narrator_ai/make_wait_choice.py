# actors/narrator/narrator_ai/make_wait_choice.py
from __future__ import annotations

from typing import Dict, Any, TYPE_CHECKING

from .types import NarrationChoice

if TYPE_CHECKING:  # 型チェック用。実行時には循環 import を避ける。
    from .narrator_ai import NarratorAI


def make_wait_choice_impl(
    self: "NarratorAI",
    world_state: Dict[str, Any] | None = None,
    floria_state: Dict[str, Any] | None = None,
) -> NarrationChoice:
    """
    「何もしない」。
    プレイヤー一人（party_mode == "alone"）のときは、
    ここでも「誰もいないのでここでじっとしているしかない」旨を返す。
    """
    snap = self._get_scene_snapshot()  # type: ignore[attr-defined]
    party_mode = snap["party_mode"]

    if party_mode == "alone":
        speak = (
            "この場には他に誰の気配もない。"
            "あなたはしばし足を止め、静かな空気の中で次の出会いを待つことにした。"
        )
    else:
        intent = "何もせず、しばらく黙って様子を見る"
        speak = self._refine(intent_text=intent, label="wait")  # type: ignore[attr-defined]

    return NarrationChoice(
        kind="wait",
        label="何もしない",
        speak_text=speak,
    )
