# actors/emotion_modes/debate_selector.py
from __future__ import annotations

from typing import Optional

from .base_selector import BaseModeSelector


class DebateModeSelector(BaseModeSelector):
    """
    「討論モード」を判定する Selector。

    ・怒り（anger）が高い
    ・あるいは緊張（tension）が高い
    ・あるいは興奮が高いが arousal が低い（白熱した議論）

    のいずれかを満たせば "debate" を返す。
    """
    name = "debate"
    priority = 80

    def select(self, signal: JudgeSignal) -> Optional[str]:
        # 1) 怒りが高い
        if signal.anger >= 0.50:
            return "debate"

        # 2) 緊張が高い
        if signal.tension >= 0.65:
            return "debate"

        # 3) 興奮は高いが arousal は低い → 白熱した議論寄り
        if signal.excitement >= 0.70 and signal.arousal < 0.30:
            return "debate"

        return None

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
