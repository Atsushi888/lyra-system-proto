# actors/judge_strategy/judge_strategy_erotic.py
from __future__ import annotations

from typing import Dict, Any, Tuple

from actors.judge_strategy.judge_types import (
    BaseJudgeStrategy,
    JudgeCandidate,
    JudgeContext,
)


class EroticJudgeStrategy(BaseJudgeStrategy):
    """
    エロエロモード用の暫定実装。
    とりあえず「甘い/親密ワードのヒット数」で差を付ける。
    """

    mode_name = "erotic"

    def score_candidate(
        self,
        cand: JudgeCandidate,
        context: JudgeContext,
    ) -> Tuple[float, Dict[str, Any]]:

        text = cand.text or ""
        length = len(text)

        keywords = [
            "キス", "唇", "胸", "肌", "抱き", "頬", "熱い",
            "ドキドキ", "とろけ", "鼓動", "吐息", "寄せ",
            "絡め", "触れ", "触れ合", "甘い", "蕩け",
        ]
        hits = sum(text.count(k) for k in keywords)

        base_score = max(0.1, length ** 0.5 / 20.0)
        erotic_score = hits * 0.8
        score = base_score + erotic_score

        details: Dict[str, Any] = {
            "length": length,
            "erotic_hits": hits,
            "base_score": base_score,
            "erotic_score": erotic_score,
        }
        return float(score), details
