# actors/models_ai.py

from __future__ import annotations

from typing import Dict, Any, List

from llm.llm_manager import LLMManager


class ModelsAI:
    """
    複数の LLM モデルから回答を収集するクラス。

    - LLMManager に登録されているモデル一覧を見て、
      enabled なものだけ順番に呼び出す。
    - 実際の API 呼び出しは LLMManager.call_model() に委譲する。
    """

    def __init__(self, llm_manager: LLMManager) -> None:
        # persona ごとに共有されている LLMManager
        self.llm_manager = llm_manager

    def _normalize_result(self, result: Any) -> Dict[str, Any]:
        """
        LLMManager.call_model() の戻り値を標準化する。

        - 文字列だけ → text に入れる
        - (text, usage, ...) のタプル → text / usage を取り出す
        """
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

        戻り値の例:
            {
              "gpt4o": {
                "status": "ok",
                "text": "...",
                "usage": {...} or None,
                "meta": {...},
              },
              "gpt51": {...},
              "hermes": {...},
            }
        """
        results: Dict[str, Any] = {}

        # LLMManager から model_props を取得
        model_props = self.llm_manager.get_model_props()

        for name, props in model_props.items():
            enabled = props.get("enabled", True)
            if not enabled:
                results[name] = {
                    "status": "disabled",
                    "error": "disabled_by_config",
                }
                continue

            try:
                # 実際の呼び出しは LLMManager に委譲
                raw = self.llm_manager.call_model(
                    model_name=name,
                    messages=messages,
                )
                norm = self._normalize_result(raw)
                norm["status"] = "ok"
                results[name] = norm
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "error": str(e),
                }

        return results
