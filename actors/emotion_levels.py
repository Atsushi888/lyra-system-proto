# actors/emotion_levels.py
from __future__ import annotations

from typing import Literal


AffectionLevel = Literal["low", "mid", "high", "extreme"]


def affection_to_level(score: float) -> AffectionLevel:
    """
    0.0〜1.0 の好感度スコアから、段階レベルを決定する共通ヘルパ。
    Lyra 全体でこの関数を使うことで、Persona / Emotion / UI / ModelsAI を同期させる。
    """
    try:
        v = float(score)
    except Exception:
        v = 0.0

    if v < 0.33:
        return "low"
    elif v < 0.66:
        return "mid"
    elif v < 0.90:
        return "high"
    else:
        return "extreme"
