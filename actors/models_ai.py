# actors/models_ai.py

from __future__ import annotations
from typing import Dict, Any

from llm.llm_router import LLMRouter


class ModelsAI:
    """
    各AI(GPT-4o / Hermes / GPT-5.1 など)から回答を収集し、
    llm_meta["models"] に格納する責務を持つクラス。

    現段階では：
      - LLMRouter を保持
      - collect() で複数モデルの回答を取得
      - Python辞書(JSON相当の構造)で返す
    """

    def __init__(self) -> None:
        # すべてのモデル呼び出しの窓口
        self.router = LLMRouter()

    def collect(self, user_text: str) -> Dict[str, Any]:
        """
        LLMRouter を使って複数モデルから回答を取得する。

        返却形式(例):
        {
            "gpt4o": {
                "text": "...",
                "usage": {...},
                "meta": {...},
                "status": "ok"
            },
            "hermes": {...},
            "gpt51": {...}
        }
        """

        results: Dict[str, Any] = {}

        # GPT-4o
        try:
            txt, usage, meta = self.router.call_gpt4o(user_text)
            results["gpt4o"] = {
                "text": txt,
                "usage": usage,
                "meta": meta,
                "status": "ok",
            }
        except Exception as e:
            results["gpt4o"] = {
                "status": "error",
                "error": str(e),
            }

        # Hermes
        try:
            txt, usage, meta = self.router.call_hermes(user_text)
            results["hermes"] = {
                "text": txt,
                "usage": usage,
                "meta": meta,
                "status": "ok",
            }
        except Exception as e:
            results["hermes"] = {
                "status": "error",
                "error": str(e),
            }

        # GPT-5.1 (仮)
        try:
            txt, usage, meta = self.router.call_gpt51(user_text)
            results["gpt51"] = {
                "text": txt,
                "usage": usage,
                "meta": meta,
                "status": "ok",
            }
        except Exception as e:
            results["gpt51"] = {
                "status": "error",
                "error": str(e),
            }

        return results
