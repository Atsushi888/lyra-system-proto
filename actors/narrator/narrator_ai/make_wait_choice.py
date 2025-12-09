from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # 型ヒント専用（実行時には import しない）
    from .narrator_ai import NarratorAI, NarrationChoice


def make_wait_choice_impl(
    narrator: "NarratorAI",
    world_state: Dict[str, Any] | None = None,
    floria_state: Dict[str, Any] | None = None,
) -> "NarrationChoice":
    """
    「何もしない」救済アクションの共通実装。

    - party_mode == "alone" のとき:
        → その場には誰もおらず、静かに待つ描写を直接返す。
    - それ以外:
        → narrator._refine() を通してライトノベル調 1〜2文に整形する。
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

    # 遅延インポートで循環参照を回避
    from .narrator_ai import NarrationChoice

    return NarrationChoice(
        kind="wait",
        label="何もしない",
        speak_text=speak,
    )
