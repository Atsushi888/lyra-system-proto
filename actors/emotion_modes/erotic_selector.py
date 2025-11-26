# actors/emotion_modes/erotic_selector.py
from __future__ import annotations

from typing import Optional

from .base_selector import BaseModeSelector

from typing import Dict, Any, Tuple

from actors.emotion_modes.judge_types import (
    BaseJudgeStrategy,
    JudgeCandidate,
    JudgeContext,
)

class EroticModeSelector(BaseModeSelector):
    """
    「エロエロモード」を判定する Selector。

    ・性的昂揚（arousal）が高い
    ・もしくは好意＋興奮が高い

    いずれかを満たせば "erotic" を返す。
    """
    name = "erotic"
    priority = 100

    def select(self, signal: JudgeSignal) -> Optional[str]:
        # 1) 性的昂揚が高い
        if signal.arousal >= 0.55:
            return "erotic"

        # 2) 好意＋ワクワクが高い → ちょっと甘い空気
        if signal.affection >= 0.75 and signal.excitement >= 0.40:
            return "erotic"

        return None
