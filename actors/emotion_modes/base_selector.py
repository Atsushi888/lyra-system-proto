# actors/emotion_modes/base_selector.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .emotion_modes.context import JudgeSignal


class BaseModeSelector(ABC):
    """
    judge_mode を決定するための Strategy の基底クラス。

    - name:     この Selector が担当するモード名（例: "erotic"）
    - priority: 適用優先度（大きいほど先に評価される）

    select(signal) が None を返した場合は「このモードではない」扱い。
    "normal" / "erotic" / "debate" などの文字列を返した場合はそのモードを採用する。
    """

    name: str = "base"
    priority: int = 0

    @abstractmethod
    def select(self, signal: JudgeSignal) -> Optional[str]:
        """
        signal を見て、担当モードであるべきならモード名を返す。
        担当外なら None を返す。
        """
        raise NotImplementedError
