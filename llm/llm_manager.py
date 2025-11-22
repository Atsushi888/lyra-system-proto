# llm/llm_manager.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional
import os

from llm.llm_router import (
    call_gpt4o,
    call_gpt51,
    call_hermes,
    call_grok,
    call_gemini,
)


class LLMManager:
    """
    ベンダー横断 LLM 管理クラス。
    register_* でモデルを登録し、
    call_model(name, messages) で任意のモデルを呼び出す。
    """

    _POOL: Dict[str, "LLMManager"] = {}

    @classmethod
    def get_or_create(cls, persona_id: str = "default") -> "LLMManager":
        if persona_id in cls._POOL:
            return cls._POOL[persona_id]

        manager = cls(persona_id=persona_id)

        # ===== デフォルトモデル群 =====
        manager.register_gpt4o(priority=3.0, enabled=True)
        manager.register_gpt51(priority=2.0, enabled=True)
        manager.register_hermes(priority=1.0, enabled=True)

        # ===== Grok / Gemini：導入済みなので有効化 =====
        manager.register_grok(priority=1.5, enabled=True)
        manager.register_gemini(priority=1.5, enabled=True)

        cls._POOL[persona_id] = manager
        return manager

    # -----------------------------------------------------

    def __init__(self, persona_id: str = "default") -> None:
        self.persona_id = persona_id
        self._models: Dict[str, Dict[str, Any]] = {}

    # -----------------------------------------------------
    # ★ モデル登録関数
    # -----------------------------------------------------

    def register_model(
        self,
        name: str,
        vendor: str,
        router_fn,
        priority: float,
        enabled: bool,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:

        self._models[name] = {
            "vendor": vendor,
            "router_fn": router_fn,
            "priority": priority,
            "enabled": enabled,
            "extra": extra or {},
        }

    # ---- 個別登録ラッパー ----

    def register_gpt4o(self, priority=3.0, enabled=True):
        self.register_model(
            name="gpt4o",
            vendor="openai",
            router_fn=call_gpt4o,
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENAI_API_KEY", "model_family": "gpt-4o"},
        )

    def register_gpt51(self, priority=2.0, enabled=True):
        self.register_model(
            name="gpt51",
            vendor="openai",
            router_fn=call_gpt51,
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENAI_API_KEY", "model_family": "gpt-5.1"},
        )

    def register_hermes(self, priority=1.0, enabled=True):
        self.register_model(
            name="hermes",
            vendor="openrouter",
            router_fn=call_hermes,
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENROUTER_API_KEY", "model_family": "hermes"},
        )

    def register_grok(self, priority=1.5, enabled=False):
        self.register_model(
            name="grok",
            vendor="xai",
            router_fn=call_grok,
            priority=priority,
            enabled=enabled,
            extra={"env_key": "GROK_API_KEY", "model_family": "grok"},
        )

    def register_gemini(self, priority=1.5, enabled=False):
        self.register_model(
            name="gemini",
            vendor="google",
            router_fn=call_gemini,
            priority=priority,
            enabled=enabled,
            extra={"env_key": "GEMINI_API_KEY", "model_family": "gemini"},
        )

    # -----------------------------------------------------
    # ★ モデル取得系
    # -----------------------------------------------------

    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        return self._models

    def get_models_sorted(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        priority の高い順（降順）で返す。
        enabled=False のモデルは除外。
        """
        items = [
            (name, cfg)
            for name, cfg in self._models.items()
            if cfg.get("enabled", False)
        ]
        items.sort(key=lambda x: x[1]["priority"], reverse=True)
        return items

    # -----------------------------------------------------
    # ★ 呼び出し本体
    # -----------------------------------------------------

    def call_model(
        self, name: str, messages: List[Dict[str, str]], **kwargs
    ) -> Tuple[str, Dict[str, Any]]:

        if name not in self._models:
            raise ValueError(f"Unknown model name: {name}")

        cfg = self._models[name]

        if not cfg.get("enabled", False):
            raise RuntimeError(f"Model '{name}' is disabled.")

        router_fn = cfg["router_fn"]
        reply, usage = router_fn(messages=messages, **kwargs)
        return reply, usage
