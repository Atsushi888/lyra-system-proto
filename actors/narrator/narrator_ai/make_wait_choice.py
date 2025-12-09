from __future__ import annotations

from typing import Dict, Any, Optional, TYPE_CHECKING

from .narrator_ai import NarrationChoice

if TYPE_CHECKING:
    from .narrator_ai import NarratorAI


def build_wait_choice(
    narrator: "NarratorAI",
    world_state: Dict[str, Any] | None = None,
    floria_state: Dict[str, Any] | None = None,
) -> NarrationChoice:
    """
    「何もしない」。
    プレイヤー一人（party_mode == "alone"）かつ外野なしのときだけ
    固定文にして、それ以外は Refiner に投げる。
    """
    snap = narrator._get_scene_snapshot()
    party_mode = snap["party_mode"]
    others_present: bool = bool(snap.get("others_present", False))

    if party_mode == "alone" and not others_present:
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
