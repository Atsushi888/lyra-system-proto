# actors/models_ai.py

from __future__ import annotations
from typing import Dict, Any, List

from llm.llm_router import LLMRouter


class ModelsAI:
    def __init__(self) -> None:
        self.router = LLMRouter()

    def collect(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        LLMRouter を使って複数モデルから回答を取得する。

        ※ LLMRouter は messages(list[role/content]) を受け取る想定なので、
          user_text ではなく messages をそのまま渡す。
        """
        results: Dict[str, Any] = {}

        # GPT-4o
        try:
            txt, usage, meta = self.router.call_gpt4o(messages)
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
            txt, usage, meta = self.router.call_hermes(messages)
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
            txt, usage, meta = self.router.call_gpt51(messages)
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
