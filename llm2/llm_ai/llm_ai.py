# llm2/llm_ai/llm_ai.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import os

from llm2.llm_ai.llm_adapters.base import BaseLLMAdapter


@dataclass
class LLMModelConfig:
    """
    LLMAI 内で管理する1モデル分の設定。
    """
    name: str
    adapter: BaseLLMAdapter
    vendor: str = "unknown"
    priority: float = 1.0
    enabled: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)  # デフォルト呼び出しパラメータ


class LLMAI:
    """
    LLM呼び出しのターミナル。

    - register_* で Adapter を登録
    - call() で呼び出し（Adapter.call に委譲）
    - get_model_props() で UI/ModelsAI2 互換の一覧を返す
    """

    def __init__(self, persona_id: str = "default") -> None:
        self.persona_id = persona_id
        self._models: Dict[str, LLMModelConfig] = {}

    # ===========================================================
    # Adapter 登録
    # ===========================================================
    def register_adapter(
        self,
        adapter: BaseLLMAdapter,
        *,
        vendor: Optional[str] = None,
        priority: float = 1.0,
        enabled: bool = True,
        extra: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        register_* から呼ばれる入口。

        adapter.name をキーとして登録する。
        """
        name = getattr(adapter, "name", "") or ""
        if not name:
            raise ValueError("Adapter.name is empty")

        # 既知モデルは env_key 等を自動補完しておく（UIの可用性判定に使える）
        auto_vendor, auto_extra = self._infer_vendor_extra(name)

        cfg = LLMModelConfig(
            name=name,
            adapter=adapter,
            vendor=vendor or auto_vendor,
            priority=float(priority),
            enabled=bool(enabled),
            extra={**auto_extra, **(extra or {})},
            params=dict(params or {}),
        )
        self._models[name] = cfg

    @staticmethod
    def _infer_vendor_extra(model_name: str) -> Tuple[str, Dict[str, Any]]:
        """
        既知の論理名から vendor / env_key / model_family を推定。
        """
        if model_name in ("gpt51", "gpt4o"):
            return "openai", {"env_key": "OPENAI_API_KEY", "model_family": model_name}
        if model_name == "grok":
            return "xai", {"env_key": "GROK_API_KEY", "model_family": "grok"}
        if model_name == "gemini":
            return "google", {"env_key": "GEMINI_API_KEY", "model_family": "gemini"}
        if model_name in ("hermes", "hermes_new", "llama_unc"):
            return "openrouter", {
                "env_key": "OPENROUTER_API_KEY",
                "model_family": model_name,
            }
        return "unknown", {}

    # ===========================================================
    # 呼び出し
    # ===========================================================
    def call(
        self,
        *,
        model_name: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        """
        互換のため、戻り値は Adapter 実装に合わせる：
        - (text, usage) が基本
        """
        cfg = self._models.get(model_name)
        if cfg is None:
            raise ValueError(f"Unknown model: {model_name}")

        # 登録済み params をデフォルトとして合成
        call_params = dict(cfg.params)
        call_params.update(kwargs)

        return cfg.adapter.call(messages=messages, **call_params)

    # ===========================================================
    # 情報取得（互換）
    # ===========================================================
    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        """
        旧 LLMManager.get_model_props 互換。

        ModelsAI2 / UI が読むので、enabled / priority / extra / params を返す。
        """
        out: Dict[str, Dict[str, Any]] = {}
        for name, cfg in self._models.items():
            out[name] = {
                "vendor": cfg.vendor,
                "priority": cfg.priority,
                "enabled": cfg.enabled,
                "extra": dict(cfg.extra),
                "params": dict(cfg.params),

                # 互換性のために defaults も置く（古い呼び出しが残ってても壊れにくい）
                "defaults": dict(cfg.params),
            }
        return out

    def get_models_sorted(self) -> Dict[str, Dict[str, Any]]:
        items = sorted(self._models.items(), key=lambda kv: kv[1].priority, reverse=True)
        out: Dict[str, Dict[str, Any]] = {}
        for name, cfg in items:
            out[name] = {
                "vendor": cfg.vendor,
                "priority": cfg.priority,
                "enabled": cfg.enabled,
                "extra": dict(cfg.extra),
                "params": dict(cfg.params),
                "defaults": dict(cfg.params),
            }
        return out

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        env_key が存在するモデルは APIキー有無も付与して返す。
        """
        props = self.get_model_props()
        for name, p in props.items():
            extra = p.get("extra") or {}
            env_key = extra.get("env_key")
            has_key = True
            if env_key:
                has_key = bool(os.getenv(env_key, ""))
            p["has_key"] = has_key
        return props

    def set_enabled_models(self, enabled: Dict[str, bool]) -> None:
        for name, flag in enabled.items():
            if name in self._models:
                self._models[name].enabled = bool(flag)
