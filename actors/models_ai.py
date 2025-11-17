# actors/models_ai.py

from __future__ import annotations
from typing import Dict, Any


class ModelsAI:
    """
    各AI(GPT-4o / Hermes / その他モデル)から回答を収集し、
    llm_meta["models"] に格納する責務を持つクラス。

    現段階では Router を呼ばず、
    モックで「形だけ」を作る。
    """

    def __init__(self) -> None:
        pass

    def collect(self, user_text: str) -> Dict[str, Any]:
        """
        将来:
          - GPT-4o / Hermes / etc を Router 経由で叩く
          - 各結果を辞書にまとめる

        現在:
          - モック 1 モデル分だけ返す
        """
        return {
            "gpt4o": {
                "text": f"(mock) GPT-4o の回答: {user_text}",
                "status": "ok",
                "usage": None,
                "meta": {},
            },
        }
