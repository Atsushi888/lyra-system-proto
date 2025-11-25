from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

from llm.llm_adapter import (
    BaseLLMAdapter,
    GPT4oAdapter,
    GPT51Adapter,
    HermesOldAdapter,
    HermesNewAdapter,
    GrokAdapter,
    GeminiAdapter,
)


@dataclass
class LLMModelConfig:
    name: str
    vendor: str
    router_fn: str  # 互換用。実際の実行は Adapter 側に移管。
    priority: float = 1.0
    enabled: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


class LLMManager:
    _POOL: Dict[str, "LLMManager"] = {}

    @classmethod
    def get_or_create(cls, persona_id: str = "default") -> "LLMManager":
        if persona_id in cls._POOL:
            return cls._POOL[persona_id]

        manager = cls(persona_id=persona_id)

        # === Default models ===
        manager.register_gpt4o(priority=3.0, enabled=False)   # ← Multi-LLM では不参加
        manager.register_gpt51(priority=2.0, enabled=True)
        manager.register_hermes(priority=1.0, enabled=True)   # 旧 Hermes
        manager.register_grok(priority=1.5, enabled=True)
        manager.register_gemini(priority=1.5, enabled=True)
        # 新 Hermes はテスト用として別途有効化できるようにする
        manager.register_hermes_new(priority=0.5, enabled=False)

        cls._POOL[persona_id] = manager
        return manager

    def __init__(self, persona_id: str = "default") -> None:
        self.persona_id = persona_id
        self._models: Dict[str, LLMModelConfig] = {}

        # name -> Adapter インスタンス
        self._adapters: Dict[str, BaseLLMAdapter] = {}

        # EmotionAI → LLMManager 経由でモードを受け取る用
        self._last_mode: Optional[str] = None

    # ==================================================
    # モデル登録
    # ==================================================
    def register_model(
        self,
        name: str,
        *,
        vendor: str,
        router_fn: str,
        priority: float = 1.0,
        enabled: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = LLMModelConfig(
            name=name,
            vendor=vendor,
            router_fn=router_fn,
            priority=priority,
            enabled=enabled,
            extra=extra or {},
        )
        self._models[name] = cfg

    def register_gpt4o(
        self,
        *,
        priority: float = 3.0,
        enabled: bool = False,   # ← デフォルトで無効
    ) -> None:
        """
        gpt-4o-mini は Multi-LLM からは基本外す。
        個別用途で使いたい場合のみ enabled=True で再登録する想定。
        """
        self.register_model(
            "gpt4o",
            vendor="openai",
            router_fn="call_gpt4o",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENAI_API_KEY", "model_family": "gpt-4o"},
        )
        self._adapters["gpt4o"] = GPT4oAdapter()

    def register_gpt51(
        self,
        *,
        priority: float = 2.0,
        enabled: bool = True,
    ) -> None:
        self.register_model(
            "gpt51",
            vendor="openai",
            router_fn="call_gpt51",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENAI_API_KEY", "model_family": "gpt-5.1"},
        )
        self._adapters["gpt51"] = GPT51Adapter()

    def register_hermes(
        self,
        *,
        priority: float = 1.0,
        enabled: bool = True,
    ) -> None:
        """
        旧 Hermes（安定版）として "hermes" 名義で登録。
        """
        self.register_model(
            "hermes",
            vendor="openrouter",
            router_fn="call_hermes_old",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENROUTER_API_KEY", "model_family": "hermes_old"},
        )
        self._adapters["hermes"] = HermesOldAdapter()

    def register_hermes_new(
        self,
        *,
        priority: float = 0.5,
        enabled: bool = False,
    ) -> None:
        """
        新 Hermes 用の登録（デフォルトでは無効）。
        """
        self.register_model(
            "hermes_new",
            vendor="openrouter",
            router_fn="call_hermes_new",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENROUTER_API_KEY", "model_family": "hermes_new"},
        )
        self._adapters["hermes_new"] = HermesNewAdapter()

    def register_grok(
        self,
        *,
        priority: float = 1.5,
        enabled: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        extra = {
            "env_key": "GROK_API_KEY",
            "model_family": "grok-2",
        }
        if params:
            extra["params"] = params

        self.register_model(
            "grok",
            vendor="xai",
            router_fn="call_grok",
            priority=priority,
            enabled=enabled,
            extra=extra,
        )
        self._adapters["grok"] = GrokAdapter()

    def register_gemini(
        self,
        *,
        priority: float = 1.5,
        enabled: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        extra = {
            "env_key": "GEMINI_API_KEY",
            "model_family": "gemini-2.0",
        }
        if params:
            extra["params"] = params

        self.register_model(
            "gemini",
            vendor="google",
            router_fn="call_gemini",
            priority=priority,
            enabled=enabled,
            extra=extra,
        )
        self._adapters["gemini"] = GeminiAdapter()

    # ==================================================
    # 実行
    # ==================================================
    def call_model(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        cfg = self._models.get(model_name)
        if cfg is None:
            raise ValueError(f"Unknown model: {model_name}")

        adapter = self._adapters.get(model_name)
        if adapter is None:
            raise RuntimeError(f"No adapter registered for model: {model_name}")

        # EmotionAI → モードだけを引き取り、Adapter へは渡さない
        mode = kwargs.pop("mode", None)
        if mode is not None:
            self._last_mode = mode

        # extra.params があれば kwargs にマージ
        extra = cfg.extra or {}
        params = extra.get("params") or {}
        merged_kwargs = {**params, **kwargs}

        return adapter.call(messages=messages, **merged_kwargs)

    # ==================================================
    # 取得
    # ==================================================
    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = {}
        for name, cfg in self._models.items():
            result[name] = {
                "vendor": cfg.vendor,
                "router_fn": cfg.router_fn,
                "priority": cfg.priority,
                "enabled": cfg.enabled,
                "extra": dict(cfg.extra),
            }
        return result

    def get_models_sorted(self) -> Dict[str, Dict[str, Any]]:
        items = sorted(
            self._models.items(),
            key=lambda kv: kv[1].priority,
            reverse=True,
        )
        return {
            name: {
                "vendor": cfg.vendor,
                "router_fn": cfg.router_fn,
                "priority": cfg.priority,
                "enabled": cfg.enabled,
                "extra": dict(cfg.extra),
            }
            for name, cfg in items
        }
