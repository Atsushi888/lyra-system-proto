from __future__ import annotations

from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # 型ヒント用。実行時には import されないので循環参照にならない。
    from .narrator_ai import NarratorAI


def make_wait_choice_impl(
    narrator: "NarratorAI",
    *,
    world_state: Dict[str, Any] | None = None,
    floria_state: Dict[str, Any] | None = None,
) -> str:
    """
    NarratorAI.make_wait_choice() の実装本体。
    戻り値は speak_text のみ。
    """
    snap = narrator._get_scene_snapshot()
    party_mode = snap["party_mode"]

    if party_mode == "alone":
        # プレイヤーだけ。
        return (
            "この場には他に誰の気配もない。"
            "あなたはしばし足を止め、静かな空気の中で次の出会いを待つことにした。"
        )

    # 二人（または複数人）いる場合は、Refiner に整形させる
    intent = "何もせず、しばらく黙って様子を見る"
    speak = narrator._refine(intent_text=intent, label="wait")
    return speak
