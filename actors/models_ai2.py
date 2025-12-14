from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import traceback

from llm.llm_manager import LLMManager


CompletionType = Union[Dict[str, Any], Tuple[Any, ...], str]


class ModelsAI2:
    """
    複数 LLM から一斉に回答案を集めるためのクラス（デバッグ強化版）。

    ★ 方針
    - 例外が出ても「models が空」になることを絶対に防ぐ
    - 各モデルごとに status / error / traceback を必ず残す
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        *,
        enabled_models: Optional[List[str]] = None,
    ) -> None:
        self.llm_manager = llm_manager
        self.model_props: Dict[str, Dict[str, Any]] = llm_manager.get_model_props()

        if enabled_models is not None:
            self.enabled_models = list(enabled_models)
        else:
            self.enabled_models = [
                name for name, props in self.model_props.items()
                if props.get("enabled", True)
            ]

    # ---------------------------------------
    # 内部ヘルパ：LLM からの戻り値を正規化
    # ---------------------------------------
    @staticmethod
    def _normalize_completion(completion: CompletionType) -> Dict[str, Any]:
        text: str = ""
        usage: Any = None
        raw: Any = None

        if isinstance(completion, dict):
            raw = completion
            text = (
                completion.get("text")
                or completion.get("content")
                or completion.get("message")
                or ""
            )
            usage = completion.get("usage")

        elif isinstance(completion, (tuple, list)):
            raw = {"raw_tuple": completion}
            if completion:
                text = str(completion[0] or "")
            if len(completion) >= 2:
                usage = completion[1]

        else:
            raw = {"raw": completion}
            text = "" if completion is None else str(completion)

        return {"text": text, "usage": usage, "raw": raw}

    # ---------------------------------------
    # メイン
    # ---------------------------------------
    def collect(
        self,
        messages: List[Dict[str, str]],
        *,
        mode_current: str = "normal",
        emotion_override: Optional[Dict[str, Any]] = None,
        reply_length_mode: str = "auto",
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        if not self.enabled_models:
            # ★ ここが重要：enabled_models 空でも必ず痕跡を残す
            results["_system"] = {
                "status": "error",
                "text": "",
                "error": "enabled_models is empty",
                "traceback": None,
            }
            return results

        for model_name in self.enabled_models:
            try:
                completion: CompletionType = self.llm_manager.chat(
                    model=model_name,
                    messages=messages,
                )

                norm = self._normalize_completion(completion)

                results[model_name] = {
                    "status": "ok",
                    "text": norm["text"],
                    "raw": norm["raw"],
                    "usage": norm["usage"],
                    "error": None,
                    "traceback": None,
                    "mode_current": mode_current,
                    "emotion_override": emotion_override,
                    "reply_length_mode": reply_length_mode,
                }

            except Exception as e:
                results[model_name] = {
                    "status": "error",
                    "text": "",
                    "raw": None,
                    "usage": None,
                    "error": str(e),
                    "traceback": traceback.format_exc(limit=6),
                    "mode_current": mode_current,
                    "emotion_override": emotion_override,
                    "reply_length_mode": reply_length_mode,
                }

        return results
