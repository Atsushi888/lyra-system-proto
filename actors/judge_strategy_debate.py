# actors/judge_strategy_debate.py
from __future__ import annotations

from typing import Dict, Any, Tuple
from actors.judge_types import (
    BaseJudgeStrategy,
    JudgeCandidate,
    JudgeContext,
)


class DebateJudgeStrategy(BaseJudgeStrategy):
    """
    討論・論理重視モード。
    接続詞や因果関係ワードの出現数をスコア化。
    """

    mode_name = "debate"

    def score_candidate(
        self,
        cand: JudgeCandidate,
        context: JudgeContext,
    ) -> Tuple[float, Dict[str, Any]]:

        text = cand.text or ""
        length = len(text)

        logic_words = [
            "しかし", "一方で", "なぜなら", "つまり",
            "結論として", "まず", "次に", "最後に",
            "ゆえに", "したがって", "だから", "とはいえ",
        ]

        hits = sum(text.count(k) for k in logic_words)

        base_score = length ** 0.5
        logic_score = hits * 3.0

        score = base_score + logic_score

        details = {
            "length": length,
            "base_score": base_score,
            "logic_hits": hits,
            "logic_score": logic_score,
        }
        return score, details
