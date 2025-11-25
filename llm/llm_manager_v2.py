# llm/llm_manager_v2.py
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
class LLMModelConfigV2:
    """
    1つの LLM モデルに関する設定情報（adapter 版）。

    - extra: env_key や model_family などのメタ情報
    - params: temperature / top_p / system_prompt などのデフォルト値
    """
    name: str
    vendor: str
    priority: float = 1.0
    enabled: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


class LLMManagerV2:
    """
    adapter ベースの LLM 呼び出しマネージャ。

    旧 LLMManager（router 版）とは独立して動かす想定。
    call_model() は adapter にそのまま委譲し、
    (text, usage_dict or None) を返す。
    """

    _POOL: Dict[str, "LLMManagerV2"] = {}

    # ==========================================
    # プール管理
    # ==========================================
    @classmethod
    def get_or_create(cls, persona_id: str = "default") -> "LLMManagerV2":
        if persona_id in cls._POOL:
            return cls._POOL[persona_id]

        manager = cls(persona_id=persona_id)

        # === デフォルト登録 ===
        # gpt4o は「常時登録するが、会話システムでは使わない」運用なので enabled=True のまま
        manager.register_gpt4o(priority=3.0, enabled=True)
        manager.register_gpt51(priority=2.0, enabled=True)
        manager.register_hermes(priority=1.0, enabled=True)        # 旧 Hermes
        manager.register_grok(priority=1.5, enabled=True)
        manager.register_gemini(priority=1.5, enabled=True)
        # 新 Hermes はテスト用としてひとまず無効
        manager.register_hermes_new(priority=0.5, enabled=False)

        cls._POOL[persona_id] = manager
        return manager

    # ==========================================
    # 初期化
    # ==========================================
    def __init__(self, persona_id: str = "default") -> None:
        self.persona_id = persona_id

        # name -> config
        self._models: Dict[str, LLMModelConfigV2] = {}
        # name -> Adapter インスタンス
        self._adapters: Dict[str, BaseLLMAdapter] = {}

        # Emotion / Judge 側から渡される mode を記録だけしておく
        self._last_mode: Optional[str] = None

    # ==========================================
    # モデル登録（共通）
    # ==========================================
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
    ) -> None:
        cfg = LLMModelConfigV2(
            name=name,
            vendor=vendor,
            priority=priority,
            enabled=enabled,
            extra=extra or {},
            params=params or {},
        )
        self._models[name] = cfg

        if adapter is not None:
            self._adapters[name] = adapter

    # ---------- 個別登録ヘルパ ----------

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
            adapter=GPT4oAdapter(),
        )

    def register_gpt51(
        self,
        *,
        priority: float = 2.0,
        enabled: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.register_model(
            "gpt51",
            vendor="openai",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENAI_API_KEY", "model_family": "gpt-5.1"},
            params=params,
            adapter=GPT51Adapter(),
        )

    def register_hermes(
        self,
        *,
        priority: float = 1.0,
        enabled: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        旧 Hermes（OpenRouter）を "hermes" 名義で登録。
        """
        self.register_model(
            "hermes",
            vendor="openrouter",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENROUTER_API_KEY", "model_family": "hermes_old"},
            params=params,
            adapter=HermesOldAdapter(),
        )

    def register_hermes_new(
        self,
        *,
        priority: float = 0.5,
        enabled: bool = False,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        新 Hermes（hermes-4-70b）を "hermes_new" 名義で登録。
        """
        self.register_model(
            "hermes_new",
            vendor="openrouter",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENROUTER_API_KEY", "model_family": "hermes_new"},
            params=params,
            adapter=HermesNewAdapter(),
        )

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
        self.register_model(
            "grok",
            vendor="xai",
            priority=priority,
            enabled=enabled,
            extra=extra,
            params=params,
            adapter=GrokAdapter(),
        )

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
        self.register_model(
            "gemini",
            vendor="google",
            priority=priority,
            enabled=enabled,
            extra=extra,
            params=params,
            adapter=GeminiAdapter(),
        )

    # ==========================================
    # 実際の呼び出し
    # ==========================================
    def call_model(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        """
        Adapter に処理を委譲して (text, usage_dict or None) を返す。
        """
        cfg = self._models.get(model_name)
        if cfg is None:
            raise ValueError(f"Unknown model: {model_name}")

        adapter = self._adapters.get(model_name)
        if adapter is None:
            raise RuntimeError(f"No adapter registered for model: {model_name}")

        # Emotion / Judge から渡ってきたモードは記録だけしておく
        mode = kwargs.pop("mode", None)
        if mode is not None:
            self._last_mode = mode

        # cfg.params のデフォルト値を統合
        merged_kwargs = dict(cfg.params)
        merged_kwargs.update(kwargs)

        # adapter.call() は (text, usage_dict or None) を返す想定
        return adapter.call(messages=messages, **merged_kwargs)

    # ==========================================
    # 情報取得系（UI 用）
    # ==========================================
    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        """
        モデル名 -> メタ情報 dict
        """
        result: Dict[str, Dict[str, Any]] = {}
        for name, cfg in self._models.items():
            result[name] = {
                "vendor": cfg.vendor,
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
                "priority": cfg.priority,
                "enabled": cfg.enabled,
                "extra": dict(cfg.extra),
                "params": dict(cfg.params),
            }
        return result

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        env_key が入っているかどうかも含めて返す。
        （旧 LLMManager と同じインターフェース）
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
        """
        UI からの on/off 操作用。
        """
        for name, cfg in self._models.items():
            if name in enabled:
                cfg.enabled = bool(enabled[name])
