# actors/models_ai.py

from __future__ import annotations
from typing import Dict, Any
from llm.llm_router import LLMRouter


class ModelsAI:
    """
    各AI(GPT-4o / Hermes / その他モデル)から回答を収集し、
    llm_meta["models"] に格納する責務を持つクラス。
    """

    def __init__(self) -> None:
        # Routerインスタンスを持つ（今後はここが全モデル呼び出しの中枢）
        self.router = LLMRouter()

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
