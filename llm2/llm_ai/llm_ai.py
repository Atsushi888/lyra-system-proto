# llm2/llm_ai/llm_ai.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Adapter 基底
from llm2.llm_ai.llm_adapters.base import BaseLLMAdapter


class LLMAI:
    """
    次世代 LLM 管理中枢クラス（llm2 系）。

    役割:
    - 各 LLM Adapter の登録・管理
    - models_ai2 / AnswerTalker などからの唯一の LLM 呼び出し窓口
    - パラメータ管理・デフォルト値の集約点

    設計思想:
    - 「どの LLM をどう呼ぶか」はすべてここに集約
    - Adapter は *純粋に API を叩くだけ*
    """

    def __init__(self, persona_id: str = "default") -> None:
        self.persona_id = persona_id

        # name -> adapter
        self._adapters: Dict[str, BaseLLMAdapter] = {}

        # name -> enabled
        self._enabled: Dict[str, bool] = {}

        # name -> priority
        self._priority: Dict[str, float] = {}

        # name -> default params
        self._defaults: Dict[str, Dict[str, Any]] = {}

    # ======================================================
    # Register
    # ======================================================
    def register(
        self,
        adapter: BaseLLMAdapter,
        *,
        enabled: bool = True,
        priority: float = 1.0,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Adapter を登録する。

        Parameters
        ----------
        adapter:
            BaseLLMAdapter 派生クラス
        enabled:
            初期有効状態
        priority:
            Judge / UI 用の優先度
        defaults:
            call 時に暗黙で適用されるパラメータ
        """
        name = adapter.name
        if not name:
            raise ValueError("Adapter.name must not be empty")

        self._adapters[name] = adapter
        self._enabled[name] = bool(enabled)
        self._priority[name] = float(priority)
        self._defaults[name] = defaults or {}

    # ======================================================
    # Call
    # ======================================================
    def call(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        単一モデル呼び出し。

        戻り値:
            (text, usage or None)
        """
        adapter = self._adapters.get(model)
        if adapter is None:
            raise ValueError(f"Unknown LLM model: {model}")

        if not self._enabled.get(model, False):
            raise RuntimeError(f"LLM model '{model}' is disabled")

        call_kwargs = dict(self._defaults.get(model, {}))
        call_kwargs.update(kwargs)

        return adapter.call(messages=messages, **call_kwargs)

    # OpenAI 互換名
    chat = call

    # ======================================================
    # Introspection
    # ======================================================
    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        """
        ModelsAI2 / UI 用のメタ情報。
        """
        result: Dict[str, Dict[str, Any]] = {}
        for name, adapter in self._adapters.items():
            result[name] = {
                "enabled": self._enabled.get(name, False),
                "priority": self._priority.get(name, 0.0),
                "defaults": dict(self._defaults.get(name, {})),
                "adapter": adapter.__class__.__name__,
            }
        return result

    def get_models_sorted(self) -> Dict[str, Dict[str, Any]]:
        items = sorted(
            self.get_model_props().items(),
            key=lambda kv: kv[1]["priority"],
            reverse=True,
        )
        return dict(items)

    # ======================================================
    # Enable / Disable
    # ======================================================
    def set_enabled(self, model: str, enabled: bool) -> None:
        if model not in self._adapters:
            raise ValueError(f"Unknown LLM model: {model}")
        self._enabled[model] = bool(enabled)

    def set_enabled_bulk(self, flags: Dict[str, bool]) -> None:
        for name, flag in flags.items():
            if name in self._adapters:
                self._enabled[name] = bool(flag)
