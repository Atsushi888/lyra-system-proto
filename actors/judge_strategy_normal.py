# actors/judge_strategy_normal.py
from __future__ import annotations

from typing import Dict, Any, Tuple
from actors.judge_types import (
    BaseJudgeStrategy,
    JudgeCandidate,
    JudgeContext,
)


class NormalJudgeStrategy(BaseJudgeStrategy):
    """
    通常モード（JudgeAI2 相当のロジックを移植予定）。
    現時点ではダミー実装。
    """

    mode_name = "normal"

    def score_candidate(
        self,
        cand: JudgeCandidate,
        context: JudgeContext,
    ) -> Tuple[float, Dict[str, Any]]:

        text = cand.text or ""
        length = len(text)

        # 仮ロジック：文章の長さ + 句読点の多さを軽く評価
        length_score = length ** 0.5
        richness_score = text.count("、") + text.count("。")

        score = length_score + 0.1 * richness_score

        details = {
            "length": length,
            "length_score": length_score,
            "richness_score": richness_score,
        }
        return score, details
