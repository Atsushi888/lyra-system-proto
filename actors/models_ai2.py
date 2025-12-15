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
    - Persona 由来の request params を注入できる
    - モデルの supported_parameters があれば、それで安全にフィルタして渡す
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        *,
        enabled_models: Optional[List[str]] = None,
        persona: Any = None,
    ) -> None:
        self.llm_manager = llm_manager
        self.persona = persona
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
    # Persona params
    # ---------------------------------------
    def _get_persona_request_params(self, model_name: str) -> Dict[str, Any]:
        p = self.persona
        if p is None:
            return {}
        fn = getattr(p, "get_llm_request_params", None)
        if callable(fn):
            try:
                params = fn(model_name)
                return params if isinstance(params, dict) else {}
            except Exception:
                return {}
        return {}

    # ---------------------------------------
    # supported_parameters で安全フィルタ
    # ---------------------------------------
    def _filter_request_params(self, model_name: str, params: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        model_props[model_name]["supported_parameters"] が list[str] であれば、
        params のトップレベルキーをそこに含まれるものだけ通す。

        返り値: (accepted, rejected)
        """
        if not params:
            return {}, {}

        p = self.model_props.get(model_name, {}) or {}
        supported = p.get("supported_parameters")

        if not isinstance(supported, list) or not supported:
            # supported が分からないなら、全通し（互換優先）
            return dict(params), {}

        supported_set = {str(x) for x in supported}

        accepted: Dict[str, Any] = {}
        rejected: Dict[str, Any] = {}
        for k, v in params.items():
            if str(k) in supported_set:
                accepted[k] = v
            else:
                rejected[k] = v

        return accepted, rejected

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
            results["_system"] = {
                "status": "error",
                "text": "",
                "error": "enabled_models is empty",
                "traceback": None,
            }
            return results

        for model_name in self.enabled_models:
            persona_params = self._get_persona_request_params(model_name)
            req_params, ignored_params = self._filter_request_params(model_name, persona_params)

            try:
                completion: CompletionType = self.llm_manager.chat(
                    model=model_name,
                    messages=messages,
                    **req_params,
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
                    "request_params": req_params,
                    "ignored_params": ignored_params,
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
                    "request_params": req_params,
                    "ignored_params": ignored_params,
                }

        return results
