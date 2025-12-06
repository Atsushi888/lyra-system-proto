# actors/emotion_modes/erotic_selector.py
from __future__ import annotations

from typing import Optional, Dict, Any, Tuple
from .base_selector import BaseModeSelector
from .judge_types import (
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
