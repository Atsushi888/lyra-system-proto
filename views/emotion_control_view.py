# views/emotion_control_view.py
from __future__ import annotations

from components.emotion_control import EmotionControl


class EmotionControlView:
    """
    画面表示専用の薄いラッパ。
    - 内部で EmotionControl を生成し、その render() を呼ぶだけ。
    """

    def __init__(self) -> None:
        self._ctrl = EmotionControl()

    def render(self) -> None:
        self._ctrl.render()


def create_emotion_control_view() -> EmotionControlView:
    """
    ModeSwitcher などから呼び出すためのファクトリ関数。
    """
        # 単純にインスタンスを返すだけ
    return EmotionControlView()
