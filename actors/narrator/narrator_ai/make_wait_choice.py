# actors/narrator/narrator_ai/make_wait_choice.py
from __future__ import annotations

from typing import Dict, Any

from actors/narrator/narrator_ai/narrator_ai import NarratorAI, NarrationChoice


def make_wait_choice_impl(
    narrator: NarratorAI,
    world_state: Dict[str, Any] | None = None,
    floria_state: Dict[str, Any] | None = None,
) -> NarrationChoice:
    """
    「何もしない」。
    プレイヤー一人（party_mode == "alone"）のときは、
    「誰もいないのでここでじっとしているしかない」旨を返す。
    """
    snap = narrator._get_scene_snapshot()
    party_mode = snap["party_mode"]

    if party_mode == "alone":
        speak = (
            "この場には他に誰の気配もない。"
            "あなたはしばし足を止め、静かな空気の中で次の出会いを待つことにした。"
        )
    else:
        intent = "何もせず、しばらく黙って様子を見る"
        speak = narrator._refine(intent_text=intent, label="wait")

    return NarrationChoice(
        kind="wait",
        label="何もしない",
        speak_text=speak,
    )
