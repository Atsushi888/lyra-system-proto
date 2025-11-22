# llm/llm_manager.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

from llm.llm_router import LLMRouter


@dataclass
class LLMModelConfig:
    """
    1つの LLM モデルに関する設定情報。

    - params: AI内部パラメータ（temperature/top_p/system_prompt 等）
    """
    name: str
    vendor: str
    router_fn: str
    priority: float = 1.0
    enabled: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)  # ★追加！


class LLMManager:
    _POOL: Dict[str, "LLMManager"] = {}

    @classmethod
    def get_or_create(cls, persona_id: str = "default") -> "LLMManager":
        if persona_id in cls._POOL:
            return cls._POOL[persona_id]

        manager = cls(persona_id=persona_id)

        # ★標準のモデルを登録（設定パラメータは未指定→デフォルト）
        manager.register_gpt4o(priority=3.0, enabled=True)
        manager.register_gpt51(priority=2.0, enabled=True)
        manager.register_hermes(priority=1.0, enabled=True)

        # Grok / Gemini はデフォルト disabled
        # manager.register_grok(priority=1.5, enabled=False)
        # manager.register_gemini(priority=1.5, enabled=False)

        cls._POOL[persona_id] = manager
        return manager

    def __init__(self, persona_id: str = "default") -> None:
        self.persona_id = persona_id
        self._models: Dict[str, LLMModelConfig] = {}
        self._router = LLMRouter()

    # ===========================================================
    # モデル登録
    # ===========================================================
    def register_model(
        self,
        name: str,
        *,
        vendor: str,
        router_fn: str,
        priority: float = 1.0,
        enabled: bool = True,
        extra: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,   # ★追加
    ) -> None:
        cfg = LLMModelConfig(
            name=name,
            vendor=vendor,
            router_fn=router_fn,
            priority=priority,
            enabled=enabled,
            extra=extra or {},
            params=params or {},  # ★追加
        )
        self._models[name] = cfg

    # ---- 各モデルのヘルパー ----

    def register_gpt4o(self, *, priority: float = 3.0, enabled: bool = True,
                       params: Optional[Dict[str, Any]] = None) -> None:
        self.register_model(
            "gpt4o",
            vendor="openai",
            router_fn="call_gpt4o",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENAI_API_KEY", "model_family": "gpt-4o"},
            params=params,
        )

    def register_gpt51(self, *, priority: float = 2.0, enabled: bool = True,
                       params: Optional[Dict[str, Any]] = None) -> None:
        self.register_model(
            "gpt51",
            vendor="openai",
            router_fn="call_gpt51",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENAI_API_KEY", "model_family": "gpt-5.1"},
            params=params,
        )

    def register_hermes(self, *, priority: float = 1.0, enabled: bool = True,
                        params: Optional[Dict[str, Any]] = None) -> None:
        self.register_model(
            "hermes",
            vendor="openrouter",
            router_fn="call_hermes",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENROUTER_API_KEY", "model_family": "hermes"},
            params=params,
        )

    def register_grok(self, *, priority: float = 1.5, enabled: bool = True,
                      params: Optional[Dict[str, Any]] = None) -> None:
        self.register_model(
            "grok",
            vendor="xai",
            router_fn="call_grok",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "GROK_API_KEY", "model_family": "grok-2"},
            params=params,
        )

    def register_gemini(self, *, priority: float = 1.5, enabled: bool = True,
                        params: Optional[Dict[str, Any]] = None) -> None:
        self.register_model(
            "gemini",
            vendor="google",
            router_fn="call_gemini",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "GEMINI_API_KEY", "model_family": "gemini-2.0"},
            params=params,
        )

    # ===========================================================
    # 実際の呼び出し
    # ===========================================================
    def call_model(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        cfg = self._models.get(model_name)
        if cfg is None:
            raise ValueError(f"Unknown model: {model_name}")

        fn = getattr(self._router, cfg.router_fn, None)
        if fn is None:
            raise AttributeError(
                f"LLMRouter has no method '{cfg.router_fn}' for model '{model_name}'"
            )

        # ★登録された params を引数に統合
        call_params = dict(cfg.params)
        call_params.update(kwargs)

        return fn(messages=messages, **call_params)

    # ===========================================================
    # 情報取得メソッド（ほぼ既存）
    # ===========================================================
    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for name, cfg in self._models.items():
            result[name] = {
                "vendor": cfg.vendor,
                "router_fn": cfg.router_fn,
                "priority": cfg.priority,
                "enabled": cfg.enabled,
                "extra": dict(cfg.extra),
                "params": dict(cfg.params),  # ★追加
            }
        return result

    def get_models_sorted(self) -> Dict[str, Dict[str, Any]]:
        items = sorted(self._models.items(), key=lambda kv: kv[1].priority, reverse=True)
        result: Dict[str, Dict[str, Any]] = {}
        for name, cfg in items:
            result[name] = {
                "vendor": cfg.vendor,
                "router_fn": cfg.router_fn,
                "priority": cfg.priority,
                "enabled": cfg.enabled,
                "extra": dict(cfg.extra),
                "params": dict(cfg.params),  # ★追加
            }
        return result

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        import os
        try:
            import streamlit as st
            secrets = st.secrets
        except Exception:
            secrets = {}

        props = self.get_model_props()

        for name, p in props.items():
            extra = p.get("extra") or {}
            env_key = extra.get("env_key")
            has_key = True
            if env_key:
                secret_val = ""
                if isinstance(secrets, dict):
                    secret_val = secrets.get(env_key, "")
                has_key = bool(os.getenv(env_key) or secret_val)
            p["has_key"] = has_key

        return props

    def set_enabled_models(self, enabled: Dict[str, bool]) -> None:
        for name, cfg in self._models.items():
            if name in enabled:
                cfg.enabled = bool(enabled[name])

    # ===========================================================
    # YAML（既存）
    # ===========================================================
    def load_default_config(self, path: Optional[str] = None) -> bool:
        import os
        if path is None:
            path = "llm_default.yaml"
        if not os.path.exists(path):
            return False

        try:
            import yaml
        except Exception:
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return False

        models = data.get("models")
        if not isinstance(models, dict):
            return False

        for name, cfg in models.items():
            if not isinstance(cfg, dict):
                continue

            vendor = str(cfg.get("vendor", ""))
            router_fn = str(cfg.get("router_fn", ""))
            if not vendor or not router_fn:
                continue

            priority_raw = cfg.get("priority", 1.0)
            try:
                priority = float(priority_raw)
            except Exception:
                priority = 1.0

            enabled = bool(cfg.get("enabled", True))
            extra = cfg.get("extra") or {}
            if not isinstance(extra, dict):
                extra = {}

            params = cfg.get("params") or {}
            if not isinstance(params, dict):
                params = {}

            self.register_model(
                name,
                vendor=vendor,
                router_fn=router_fn,
                priority=priority,
                enabled=enabled,
                extra=extra,
                params=params,
            )

        return True
