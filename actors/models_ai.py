# actors/models_ai.py

from __future__ import annotations
from typing import Dict, Any, List

from llm.llm_router import LLMRouter


class ModelsAI:
    def __init__(self, model_props: Dict[str, Dict[str, Any]]) -> None:
        self.router = LLMRouter()
        # AnswerTalker から渡されたモデル定義
        self.model_props = model_props or {}

    def _normalize_result(self, result: Any) -> Dict[str, Any]:
        reply_text = None
        usage = None
        meta: Dict[str, Any] = {}

        if isinstance(result, tuple):
            if len(result) >= 2:
                reply_text, usage = result[:2]
            elif len(result) == 1:
                reply_text = result[0]
        else:
            reply_text = result

        return {
            "text": (reply_text or ""),
            "usage": usage,
            "meta": meta,
        }

    def collect(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        for name, props in self.model_props.items():
            enabled = props.get("enabled", True)
            if not enabled:
                results[name] = {
                    "status": "disabled",
                    "error": "disabled_by_config",
                }
                continue

            fn_name = props.get("router_fn")
            if not fn_name:
                results[name] = {
                    "status": "error",
                    "error": "router_fn not defined",
                }
                continue

            fn = getattr(self.router, fn_name, None)
            if fn is None:
                results[name] = {
                    "status": "error",
                    "error": f"router has no '{fn_name}'",
                }
                continue

            try:
                raw = fn(messages)  # type: ignore[misc]
                norm = self._normalize_result(raw)
                norm["status"] = "ok"
                results[name] = norm
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e),
                }

        return results
