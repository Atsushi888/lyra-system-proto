# actors/composer_ai.py

from __future__ import annotations

from typing import Any, Dict

from actors.answer_talker import AnswerTalker


class ComposerAI(AnswerTalker):
    """
    AnswerTalker を継承した整形モジュール。

    将来的な責務:
      - llm_meta["models"], llm_meta["judge"] を参照し、
        プレイヤーに返す最終文面を組み立て、
        llm_meta["composer"] に格納する。
    """

    def __init__(self) -> None:
        super().__init__()
        # Composer 固有のパラメータがあればここへ

    # いずれ:
    # def compose(self, judge_result: Dict[str, Any]) -> str:
    #     ...
