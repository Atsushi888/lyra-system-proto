# actors/models_ai.py

from __future__ import annotations
from typing import Dict, Any, List

from llm.llm_router import LLMRouter


class ModelsAI:
    def __init__(self) -> None:
        self.router = LLMRouter()

    def _normalize_result(self, result: Any) -> Dict[str, Any]:
        """
        LLMRouter の戻り値を
        {text, usage, meta} 形式に正規化する。

        - 文字列だけ
        - (text,)
        - (text, usage)
        - (text, usage, meta)
        などに対応。
        """
        reply_text: str | None = None
        usage: Any = None
        meta: Dict[str, Any] = {}

        if isinstance(result, tuple):
            if len(result) == 3:
                reply_text, usage, meta = result
            elif len(result) == 2:
                reply_text, usage = result
            elif len(result) >= 1:
                reply_text = result[0]
        else:
            reply_text = result

        return {
            "text": reply_text or "",
            "usage": usage,
            "meta": meta,
        }

    def collect(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        LLMRouter を使って複数モデルから回答を取得する。
        戻り値は llm_meta["models"] にそのまま突っ込める dict。
        """
        results: Dict[str, Any] = {}

        # ===== GPT-4o =====
        try:
            raw = self.router.call_gpt4o(messages)
            norm = self._normalize_result(raw)
            norm["status"] = "ok"
            results["gpt4o"] = norm
        except Exception as e:
            results["gpt4o"] = {
                "status": "error",
                "error": str(e),
            }

        # ===== Hermes =====
        if hasattr(self.router, "call_hermes"):
            try:
                raw = self.router.call_hermes(messages)  # type: ignore[attr-defined]
                norm = self._normalize_result(raw)
                norm["status"] = "ok"
                results["hermes"] = norm
            except Exception as e:
                results["hermes"] = {
                    "status": "error",
                    "error": str(e),
                }
        else:
            # まだ未配線であることを明示
            results["hermes"] = {
                "status": "disabled",
                "error": "LLMRouter に call_hermes() が未実装",
            }

        # ===== GPT-5.1 (仮) =====
        if hasattr(self.router, "call_gpt51"):
            try:
                raw = self.router.call_gpt51(messages)  # type: ignore[attr-defined]
                norm = self._normalize_result(raw)
                norm["status"] = "ok"
                results["gpt51"] = norm
            except Exception as e:
                results["gpt51"] = {
                    "status": "error",
                    "error": str(e),
                }
        else:
            results["gpt51"] = {
                "status": "disabled",
                "error": "LLMRouter に call_gpt51() が未実装",
            }

        return results
