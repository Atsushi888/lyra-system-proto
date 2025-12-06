# actors/emotion_modes/normal_selector.py
from __future__ import annotations

from typing import Optional, Dict, Any, Tuple

from .base_selector import BaseModeSelector
from .judge_types import (
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

    def score_candidate(
        self,
        cand: JudgeCandidate,
        context: JudgeContext,
    ) -> Tuple[float, Dict[str, Any]]:

        text = cand.text or ""
        length = len(text)

        # 仮ロジック：
        #  - 200〜800文字くらいが気持ちよいレンジ
        ideal_min = 200
        ideal_max = 800

        if length < 50:
            score = 0.1
        elif length < ideal_min:
            score = 1.0 + (length - 50) / max(1, ideal_min - 50)
        elif length <= ideal_max:
            score = 2.0 + (length - ideal_min) / max(1, ideal_max - ideal_min)
        else:
            over = length - ideal_max
            score = max(1.0, 3.0 - over / 800.0)

        details: Dict[str, Any] = {
            "length": length,
        }
        return float(score), details

