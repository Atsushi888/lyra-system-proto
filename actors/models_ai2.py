# actors/models_ai2.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from llm.llm_manager import LLMManager


CompletionType = Union[Dict[str, Any], Tuple[Any, ...], str]


class ModelsAI2:
    """
    複数 LLM から一斉に回答案を集めるためのクラス。

    方針（LLMAI化に合わせて単純化）:
    - 「enabled なモデルを回して結果を集める」だけに責務を限定
    - 呼び出しパラメータ（defaults / temperature 等）は LLM 側（LLMAI/Registry）に集約
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
            models: List[str] = []
            for name, props in self.model_props.items():
                if props.get("enabled", True):
                    models.append(name)
            self.enabled_models = models

    # ---------------------------------------
    # 内部ヘルパ：LLM からの戻り値を正規化
    # ---------------------------------------
    @staticmethod
    def _normalize_completion(completion: CompletionType) -> Dict[str, Any]:
        """
        想定パターン:
        - dict 形式: {"text": "...", "usage": {...}, ...}
        - tuple 形式: (text, usage_dict?) など
        - str: "answer text"
        """
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
            if len(completion) >= 1:
                first = completion[0]
                text = str(first) if first is not None else ""
            if len(completion) >= 2:
                usage = completion[1]

        else:
            raw = {"raw": completion}
            text = "" if completion is None else str(completion)

        return {"text": text, "usage": usage, "raw": raw}

    # ---------------------------------------
    # メイン：全モデルから回答を集める
    # ---------------------------------------
    def collect(
        self,
        messages: List[Dict[str, str]],
        *,
        mode_current: str = "normal",
        emotion_override: Optional[Dict[str, Any]] = None,
        reply_length_mode: str = "auto",
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        messages:
            OpenAI / OpenRouter 互換の chat messages 配列。
        mode_current:
            ロギング用途（将来のプリセット切替に使える）
        emotion_override:
            ロギング用途
        reply_length_mode:
            ロギング用途
        """
        results: Dict[str, Any] = {}

        for model_name in self.enabled_models:
            try:
                # ここでは「回して集める」だけ。
                # 温度/トークン等は LLM 側（LLMAI/Registry）に集約済み前提。
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
                    "mode_current": mode_current,
                    "emotion_override": emotion_override,
                    "reply_length_mode": reply_length_mode,
                }

        return results
