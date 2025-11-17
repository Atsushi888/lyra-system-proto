# actors/judge_ai2.py

from __future__ import annotations

from typing import Any, Dict


class JudgeAI2(AnswerTalker):
    """
    AnswerTalker を継承した裁定モジュール。

    将来的な責務:
      - llm_meta["models"] を参照し、
        採用すべき候補やスコアを llm_meta["judge"] に書き込む。
    """

    def __init__(self) -> None:
        super().__init__()
        # 将来、JudgeAI2 固有の設定があればここに載せる

    # いずれ:
    # def run_judge(self, models: Dict[str, Any]) -> Dict[str, Any]:
    #     ...
