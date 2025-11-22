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
    討論モード用の暫定実装。
    接続詞や論理ワードの出現数で「論理っぽさ」を見る。
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

        base_score = max(0.1, length ** 0.5 / 20.0)
        logic_score = hits * 1.2
        score = base_score + logic_score

        details: Dict[str, Any] = {
            "length": length,
            "logic_hits": hits,
            "base_score": base_score,
            "logic_score": logic_score,
        }
        return float(score), details
