# lyra_system.py
from __future__ import annotations
from components.mode_switcher import ModeSwitcher

class LyraSystem:
    def __init__(self) -> None:
        # 初期設定・ボタン・露出・描画は全部 ModeSwitcher 側
        self.switcher = ModeSwitcher(page_title="Lyra System")

    def run(self) -> None:
        self.switcher.render()

if __name__ == "__main__":
    LyraSystem().run()
