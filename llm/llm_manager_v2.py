# llm/llm_manager_v2.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

from llm.llm_router import LLMRouter
from llm.llm_adapter import (
    BaseLLMAdapter,
    GPT4oAdapter,
    # GPT51Adapter,  # gpt51 は旧 Router 経由に戻すので使わない
    HermesOldAdapter,
    HermesNewAdapter,
    GrokAdapter,
    GeminiAdapter,
)


@dataclass
class LLMModelConfig:
    """
    1つの LLM モデルに関する設定情報。
    - adapter:  adapter 経由で呼ぶ場合に使用
    - router_fn: 旧 LLMRouter のメソッドを使う場合に使用
                 （gpt51 だけこちら）
    """

    name: str
    vendor: str
    priority: float = 1.0
    enabled: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)

    adapter: Optional[BaseLLMAdapter] = None
    router_fn: Optional[str] = None


class LLMManagerV2:
    """
    v2: adapter / router のハイブリッド版 LLMManager

    - gpt4o / hermes / grok / gemini など: llm_adapter 経由
    - gpt51: 旧 llm_router.call_gpt51 をそのまま利用
    """

    _POOL: Dict[str, "LLMManager"] = {}

    # ===========================================================
    # プール管理
    # ===========================================================
    @classmethod
    def get_or_create(cls, persona_id: str = "default") -> "LLMManager":
        if persona_id in cls._POOL:
            return cls._POOL[persona_id]

        manager = cls(persona_id=persona_id)

        # ★標準モデルを登録
        manager.register_gpt4o(priority=3.0, enabled=True)
        manager.register_gpt51(priority=2.0, enabled=True)   # ← gpt51 は router
        manager.register_hermes(priority=1.0, enabled=True)
        manager.register_grok(priority=1.5, enabled=True)
        manager.register_gemini(priority=1.5, enabled=True)

        cls._POOL[persona_id] = manager
        return manager

    # ===========================================================
    # 初期化
    # ===========================================================
    def __init__(self, persona_id: str = "default") -> None:
        self.persona_id = persona_id
        self._models: Dict[str, LLMModelConfig] = {}

        # 旧実装と互換の Router（gpt51 専用で使う）
        self._router = LLMRouter()

    # ===========================================================
    # モデル登録
    # ===========================================================
    def register_model(
        self,
        name: str,
        *,
        vendor: str,
        priority: float = 1.0,
        enabled: bool = True,
        extra: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        adapter: Optional[BaseLLMAdapter] = None,
        router_fn: Optional[str] = None,
    ) -> None:
        """
        adapter / router_fn のどちらか一方（または両方 None）を指定。
        gpt51 は router_fn="call_gpt51" を指定して登録する。
        """
        cfg = LLMModelConfig(
            name=name,
            vendor=vendor,
            priority=priority,
            enabled=enabled,
            extra=extra or {},
            params=params or {},
            adapter=adapter,
            router_fn=router_fn,
        )
        self._models[name] = cfg

    # ---- 各モデルのヘルパー ----

    def register_gpt4o(
        self,
        *,
        priority: float = 3.0,
        enabled: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.register_model(
            "gpt4o",
            vendor="openai",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENAI_API_KEY", "model_family": "gpt-4o"},
            params=params,
            adapter=GPT4oAdapter(),   # adapter 経由
            router_fn=None,
        )

    def register_gpt51(
        self,
        *,
        priority: float = 2.0,
        enabled: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        # ★ここがポイント：adapter ではなく旧 Router メソッドを使う
        self.register_model(
            "gpt51",
            vendor="openai_router",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENAI_API_KEY", "model_family": "gpt-5.1"},
            params=params,
            adapter=None,
            router_fn="call_gpt51",
        )

    def register_hermes(
        self,
        *,
        priority: float = 1.0,
        enabled: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.register_model(
            "hermes",
            vendor="openrouter",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENROUTER_API_KEY", "model_family": "hermes"},
            params=params,
            adapter=HermesOldAdapter(),
            router_fn=None,
        )

    def register_hermes_new(
        self,
        *,
        priority: float = 0.5,
        enabled: bool = False,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.register_model(
            "hermes_new",
            vendor="openrouter",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENROUTER_API_KEY", "model_family": "hermes"},
            params=params,
            adapter=HermesNewAdapter(),
            router_fn=None,
        )

    def register_grok(
        self,
        *,
        priority: float = 1.5,
        enabled: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.register_model(
            "grok",
            vendor="xai",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "GROK_API_KEY", "model_family": "grok-2"},
            params=params,
            adapter=GrokAdapter(),
            router_fn=None,
        )

    def register_gemini(
        self,
        *,
        priority: float = 1.5,
        enabled: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.register_model(
            "gemini",
            vendor="google",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "GEMINI_API_KEY", "model_family": "gemini-2.0"},
            params=params,
            adapter=GeminiAdapter(),
            router_fn=None,
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
        """
        - adapter が設定されていれば adapter.call()
        - そうでなければ router_fn で LLMRouter 経由呼び出し
        """
        cfg = self._models.get(model_name)
        if cfg is None:
            raise ValueError(f"Unknown model: {model_name}")

        # ベースのパラメータ
        call_params = dict(cfg.params)
        call_params.update(kwargs)

        # ★Emotion モード用の内部パラメータは LLM には投げない
        call_params.pop("mode", None)

        # adapter 優先
        if cfg.adapter is not None:
            return cfg.adapter.call(messages=messages, **call_params)

        # router_fn があれば旧 Router 経由
        if cfg.router_fn:
            fn = getattr(self._router, cfg.router_fn, None)
            if fn is None:
                raise AttributeError(
                    f"LLMRouter has no method '{cfg.router_fn}' for model '{model_name}'"
                )
            return fn(messages=messages, **call_params)

        raise RuntimeError(
            f"Model '{model_name}' has neither adapter nor router_fn configured."
        )

    # ===========================================================
    # 情報取得メソッド
    # ===========================================================
    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for name, cfg in self._models.items():
            result[name] = {
                "vendor": cfg.vendor,
                "priority": cfg.priority,
                "enabled": cfg.enabled,
                "extra": dict(cfg.extra),
                "params": dict(cfg.params),
                "backend": "adapter" if cfg.adapter else "router",
                "router_fn": cfg.router_fn,
            }
        return result

    def get_models_sorted(self) -> Dict[str, Dict[str, Any]]:
        items = sorted(
            self._models.items(),
            key=lambda kv: kv[1].priority,
            reverse=True,
        )
        result: Dict[str, Dict[str, Any]] = {}
        for name, cfg in items:
            result[name] = {
                "vendor": cfg.vendor,
                "priority": cfg.priority,
                "enabled": cfg.enabled,
                "extra": dict(cfg.extra),
                "params": dict(cfg.params),
                "backend": "adapter" if cfg.adapter else "router",
                "router_fn": cfg.router_fn,
            }
        return result

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        旧版と同等の has_key 判定ロジック。
        """
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
    # YAML ロード（ほぼ旧版のまま）
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
            if not vendor:
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

            # YAML からロードする場合はいったん adapter/backend は触らず、
            # register_xxx を使わずに生登録にするならここで分岐させてもよい。
            self.register_model(
                name,
                vendor=vendor,
                priority=priority,
                enabled=enabled,
                extra=extra,
                params=params,
                adapter=None,
                router_fn=None,
            )

        return True
