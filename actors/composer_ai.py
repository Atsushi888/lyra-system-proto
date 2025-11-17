# actors/composer_ai.py

from __future__ import annotations

from typing import Any, Dict


class ComposerAI:
    """
    将来的な責務:
      - llm_meta["models"], llm_meta["judge"] を参照し、
        プレイヤーに返す最終文面を組み立て、
        llm_meta["composer"] に格納する。
    """

    def __init__(self) -> None:
        super().__init__()
        # Composer 固有のパラメータがあればここへ

    # いずれ:
    def compose(self, judge_result: Dict[str, Any]) -> str:
    #     ...
