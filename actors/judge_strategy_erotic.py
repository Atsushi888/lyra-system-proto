# actors/judge_strategy_erotic.py
from __future__ import annotations

from typing import Dict, Any, Tuple
from actors.judge_types import (
    BaseJudgeStrategy,
    JudgeCandidate,
    JudgeContext,
)


class EroticJudgeStrategy(BaseJudgeStrategy):
    """
    エロエロモード専用の審査ロジック。
    （甘い/親密ワードの出現による簡易スコア）
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

        base_score = length ** 0.5
        erotic_score = hits * 2.0

        score = base_score + erotic_score

        details = {
            "length": length,
            "base_score": base_score,
            "erotic_hits": hits,
            "erotic_score": erotic_score,
        }
        return score, details
