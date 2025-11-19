# actors/models_ai.py

from __future__ import annotations

from typing import Dict, Any, List

from llm.llm_router import LLMRouter
from llm.llm_manager import LLMManager


class ModelsAI:
    """
    複数の LLM モデルから回答を収集するクラス。

    - LLMManager から model_props を取得して、
      enabled かつ available なモデルを順に呼び出す。
    """

    def __init__(self, llm_manager: LLMManager) -> None:
        self.router = LLMRouter()
        self.llm_manager = llm_manager

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
        """
        各モデルを呼び出し、結果を dict にまとめて返す。

        戻り値:
            {
              "gpt4o": {"status": "...", "text": "...", ...},
              "gpt51": {...},
              "hermes": {...},
            }
        """
        results: Dict[str, Any] = {}

        # LLMManager から model_props を取得（disabled / unavailable も含む）
        model_props = self.llm_manager.get_model_props()

        for name, props in model_props.items():
            enabled = props.get("enabled", True)
            if not enabled:
                results[name] = {
                    "status": "disabled",
                    "error": "disabled_or_unavailable",
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
