# actors/emotion_modes/context.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .base_selector import BaseModeSelector
from .erotic_selector import EroticModeSelector
from .debate_selector import DebateModeSelector
from .normal_selector import NormalModeSelector


@dataclass
class JudgeSignal:
    """
    judge_mode 決定のための「まとめた感情シグナル」。

    EmotionAI.decide_judge_mode で短期＋長期を合成した値が入る。
    Selector 側はこの値だけを見て判定すればよい。
    """
    short_mode: str = "normal"

    affection: float = 0.0
    arousal: float = 0.0
    tension: float = 0.0
    anger: float = 0.0
    sadness: float = 0.0
    excitement: float = 0.0


def get_default_selectors() -> List[BaseModeSelector]:
    """
    judge_mode 決定に使う Strategy 群を優先度順に並べて返す。

    将来、ここに TsundereModeSelector などを追加しても、
    EmotionAI 側は変更不要。
    """
    selectors: List[BaseModeSelector] = [
        EroticModeSelector(),   # erotic 判定を最優先
        DebateModeSelector(),   # 次に討論モード
        NormalModeSelector(),   # 最後に fallback
    ]
    # いちおう priority でソートしておく（大きい順）
    selectors.sort(key=lambda s: getattr(s, "priority", 0), reverse=True)
    return selectors
