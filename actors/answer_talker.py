# actors/answer_talker.py

from __future__ import annotations

from typing import Any, Dict


class AnswerTalker:
    """
    回答生成パイプラインの中核ベースクラス。

    将来的に:
      - models: 各AIからの回答収集
      - judge:  JudgeAI2 による選別
      - composer: ComposerAI による整形
    をまとめて扱う。

    現段階では:
      - llm_meta の器だけ持つ
      - speak() は何もせず reply_text をそのまま返す
    """

    def __init__(self) -> None:
        # 将来ここに llm_meta を正式に載せていく
        self.llm_meta: Dict[str, Any] = {}

    def speak(self, reply_text: str, raw_result: Any | None = None) -> str:
        """
        将来的には:
          - raw_result / llm_meta をもとに
            models → judge → composer を駆動して最終回答を返す。

        現状:
          - 何もせず、受け取った reply_text をそのまま返す。
        """
        return reply_text
