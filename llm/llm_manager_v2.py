# llm/llm_manager_v2.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

from llm.llm_router import LLMRouter
from llm.llm_adapter import (
    GPT4oAdapter,
    HermesOldAdapter,
    HermesNewAdapter,
    GrokAdapter,
    GeminiAdapter,
)


@dataclass
class LLMModelConfigV2:
    """
    1つの LLM モデルに関する設定情報（v2）。

    backend:
        "router"  … 旧 LLMRouter の call_xxx を使う
        "adapter" … llm_adapter の Adapter.call() を使う
    """

    name: str
    vendor: str
    backend: str  # "router" | "adapter"
    priority: float = 1.0
    enabled: bool = True

    # router 用
    router_fn: Optional[str] = None

    # adapter 用（self._adapters のキー）
    adapter_key: Optional[str] = None

    extra: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


class LLMManagerV2:
    """
    v2 LLM マネージャ。

    - gpt51 だけ backend="router"（旧 LLMRouter.call_gpt51 を使用）
    - それ以外（gpt4o / hermes / grok / gemini / hermes_new）は backend="adapter"
      として llm_adapter の各 Adapter を使う。
    """

    _POOL: Dict[str, "LLMManagerV2"] = {}

    # ---------------------------------------------------------
    # シングルトン取得
    # ---------------------------------------------------------
    @classmethod
    def get_or_create(cls, persona_id: str = "default") -> "LLMManagerV2":
        if persona_id in cls._POOL:
            return cls._POOL[persona_id]

        manager = cls(persona_id=persona_id)
        cls._POOL[persona_id] = manager
        return manager

    # ---------------------------------------------------------
    # 初期化
    # ---------------------------------------------------------
    def __init__(self, persona_id: str = "default") -> None:
        self.persona_id = persona_id

        # 旧実装の LLMRouter（gpt51 専用に使う）
        self._router = LLMRouter()

        # Adapter 群
        self._adapters: Dict[str, Any] = {
            "gpt4o": GPT4oAdapter(),
            # gpt51 はあえて Adapter を使わず router 経由に戻す
            "hermes": HermesOldAdapter(),
            "hermes_new": HermesNewAdapter(),
            "grok": GrokAdapter(),
            "gemini": GeminiAdapter(),
        }

        # モデル設定
        self._models: Dict[str, LLMModelConfigV2] = {}

        # デフォルト登録（必要ならあとから load_default_config で上書き可）
        self._register_defaults()

    # ---------------------------------------------------------
    # デフォルトモデル登録
    # ---------------------------------------------------------
    def _register_defaults(self) -> None:
        # gpt4o … adapter（ただし UI 側で disabled にしても OK）
        self.register_model(
            name="gpt4o",
            vendor="openai",
            backend="adapter",
            adapter_key="gpt4o",
            priority=3.0,
            enabled=True,
            extra={"env_key": "OPENAI_API_KEY", "model_family": "gpt-4o"},
            params={},
        )

        # gpt51 … ★旧 router 経由に戻す★
        self.register_model(
            name="gpt51",
            vendor="openai",
            backend="router",
            router_fn="call_gpt51",
            priority=2.0,
            enabled=True,
            extra={"env_key": "OPENAI_API_KEY", "model_family": "gpt-5.1"},
            params={},
        )

        # hermes（旧）… OpenRouter adapter
        self.register_model(
            name="hermes",
            vendor="openrouter",
            backend="adapter",
            adapter_key="hermes",
            priority=1.0,
            enabled=True,
            extra={"env_key": "OPENROUTER_API_KEY", "model_family": "hermes"},
            params={},
        )

        # hermes_new … テスト用
        self.register_model(
            name="hermes_new",
            vendor="openrouter",
            backend="adapter",
            adapter_key="hermes_new",
            priority=0.5,
            enabled=False,
            extra={"env_key": "OPENROUTER_API_KEY", "model_family": "hermes"},
            params={},
        )

        # grok … xAI adapter
        self.register_model(
            name="grok",
            vendor="xai",
            backend="adapter",
            adapter_key="grok",
            priority=1.5,
            enabled=True,
            extra={"env_key": "GROK_API_KEY", "model_family": "grok-2"},
            params={},
        )

        # gemini … Google adapter
        self.register_model(
            name="gemini",
            vendor="google",
            backend="adapter",
            adapter_key="gemini",
            priority=1.5,
            enabled=True,
            extra={"env_key": "GEMINI_API_KEY", "model_family": "gemini-2.0"},
            params={},
        )

    # ---------------------------------------------------------
    # モデル登録 API
    # ---------------------------------------------------------
    def register_model(
        self,
        name: str,
        *,
        vendor: str,
        backend: str,
        priority: float = 1.0,
        enabled: bool = True,
        router_fn: Optional[str] = None,
        adapter_key: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        backend = backend.lower()
        if backend not in ("router", "adapter"):
            raise ValueError(f"backend must be 'router' or 'adapter', got {backend}")

        cfg = LLMModelConfigV2(
            name=name,
            vendor=vendor,
            backend=backend,
            priority=priority,
            enabled=enabled,
            router_fn=router_fn,
            adapter_key=adapter_key or name,
            extra=extra or {},
            params=params or {},
        )
        self._models[name] = cfg

    # ---------------------------------------------------------
    # 実際の呼び出し
    # ---------------------------------------------------------
    def call_model(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        cfg = self._models.get(model_name)
        if cfg is None:
            raise ValueError(f"Unknown model: {model_name}")

        # モデルに設定された params をベースに、呼び出し時の kwargs で上書き
        call_params = dict(cfg.params)
        call_params.update(kwargs)

        # -----------------------------
        # router backend（gpt51 など）
        # -----------------------------
        if cfg.backend == "router":
            if not cfg.router_fn:
                raise RuntimeError(
                    f"Model '{model_name}' is router backend but router_fn is not set"
                )
            fn = getattr(self._router, cfg.router_fn, None)
            if fn is None:
                raise AttributeError(
                    f"LLMRouter has no method '{cfg.router_fn}' "
                    f"for model '{model_name}'"
                )
            # 旧 router と同様、(text, usage_dict) のタプルをそのまま返す
            return fn(messages=messages, **call_params)

        # -----------------------------
        # adapter backend
        # -----------------------------
        adapter_key = cfg.adapter_key or model_name
        adapter = self._adapters.get(adapter_key)
        if adapter is None:
            raise RuntimeError(
                f"No adapter registered for model '{model_name}' (key='{adapter_key}')"
            )

        # Adapter も (text, usage_dict) を返す設計
        return adapter.call(messages=messages, **call_params)

    # ---------------------------------------------------------
    # 情報取得メソッド
    # ---------------------------------------------------------
    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for name, cfg in self._models.items():
            result[name] = {
                "vendor": cfg.vendor,
                "backend": cfg.backend,  # ← backend 情報も見えるようにする
                "priority": cfg.priority,
                "enabled": cfg.enabled,
                "extra": dict(cfg.extra),
                "params": dict(cfg.params),
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
                "backend": cfg.backend,
                "priority": cfg.priority,
                "enabled": cfg.enabled,
                "extra": dict(cfg.extra),
                "params": dict(cfg.params),
            }
        return result

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        旧 LLMManager と互換の「利用可能モデル一覧」。
        env_key / secrets / os.environ を見て has_key を付与する。
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

    # ---------------------------------------------------------
    # YAML ロード（必要なら v1 とほぼ同じ形で上書き）
    # ---------------------------------------------------------
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
            backend = str(cfg.get("backend", "adapter")).lower()
            if not vendor or backend not in ("router", "adapter"):
                continue

            router_fn = cfg.get("router_fn")
            adapter_key = cfg.get("adapter_key") or name

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
                name=name,
                vendor=vendor,
                backend=backend,
                priority=priority,
                enabled=enabled,
                router_fn=router_fn,
                adapter_key=adapter_key,
                extra=extra,
                params=params,
            )

        return True
