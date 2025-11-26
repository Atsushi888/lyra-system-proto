# actors/emotion_modes/normal_selector.py
from __future__ import annotations

from typing import Optional

from .base_selector import BaseModeSelector

from typing import Dict, Any, Tuple

from actors.emotion_modes.judge_types import (
    BaseJudgeStrategy,
    JudgeCandidate,
    JudgeContext,
)

class NormalModeSelector(BaseModeSelector):
    """
    最後のフォールバック用 Selector。

    ここまで来た時点で「erotic でも debate でもない」ことが保証されているので、
    無条件で "normal" を返す。
    """
    name = "normal"
    priority = 0

    def select(self, signal: JudgeSignal) -> Optional[str]:
        return "normal"
