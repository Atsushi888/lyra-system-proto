# llm/llm_manager.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

from llm.llm_router import LLMRouter


@dataclass
class LLMModelConfig:
    name: str
    vendor: str
    router_fn: str
    priority: float = 1.0
    enabled: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


class LLMManager:
    _POOL: Dict[str, "LLMManager"] = {}

    @classmethod
    @classmethod
    def get_or_create(cls, persona_id: str = "default") -> "LLMManager":
        """
        persona_id ごとに LLMManager を 1 個だけ作って共有するヘルパ。

        - すでに作られていればそれを返す
        - なければ新規作成し、デフォルトモデルを登録してから保存する
        """
        if persona_id in cls._POOL:
            return cls._POOL[persona_id]

        manager = cls(persona_id=persona_id)

        # ここで標準モデルを登録（llm_default.yaml 未使用でも動く）
        manager.register_gpt4o(priority=3.0, enabled=True)
        manager.register_gpt51(priority=2.0, enabled=True)
        manager.register_hermes(priority=1.0, enabled=True)

        # Grok / Gemini もデフォルトで登録しておく（有効にするなら enabled=True）
        manager.register_grok(priority=1.5, enabled=True)
        manager.register_gemini(priority=1.5, enabled=True)

        cls._POOL[persona_id] = manager
        return manager
        
    def __init__(self, persona_id: str = "default") -> None:
        self.persona_id = persona_id
        self._models: Dict[str, LLMModelConfig] = {}
        self._router = LLMRouter()

        # ★ EmotionAI のモード受け取り保存用
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

    def register_gpt4o(self, *, priority: float = 3.0, enabled: bool = True) -> None:
        self.register_model(
            "gpt4o",
            vendor="openai",
            router_fn="call_gpt4o",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENAI_API_KEY", "model_family": "gpt-4o"},
        )

    def register_gpt51(self, *, priority: float = 2.0, enabled: bool = True) -> None:
        self.register_model(
            "gpt51",
            vendor="openai",
            router_fn="call_gpt51",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENAI_API_KEY", "model_family": "gpt-5.1"},
        )

    def register_hermes(self, *, priority: float = 1.0, enabled: bool = True) -> None:
        self.register_model(
            "hermes",
            vendor="openrouter",
            router_fn="call_hermes",
            priority=priority,
            enabled=enabled,
            extra={"env_key": "OPENROUTER_API_KEY", "model_family": "hermes"},
        )

    # ==================================================
    # 実行
    # ==================================================
    def call_model(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        """
        ModelsAI や MemoryAI から呼ばれる実行用メソッド。
        kwargs に mode が入っていても安全に処理する。
        """

        cfg = self._models.get(model_name)
        if cfg is None:
            raise ValueError(f"Unknown model: {model_name}")

        fn = getattr(self._router, cfg.router_fn, None)
        if fn is None:
            raise AttributeError(
                f"LLMRouter has no method '{cfg.router_fn}' for '{model_name}'"
            )

        # ★★ 重要：EmotionAI が渡した mode をここで受け取り、Router に渡さない
        mode = kwargs.pop("mode", None)
        if mode is not None:
            self._last_mode = mode  # 保存だけしておく

        # ★ Router には安全な kwargs だけ渡す
        return fn(messages=messages, **kwargs)

    # ==================================================
    # 取得系
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
        items = sorted(self._models.items(), key=lambda kv: kv[1].priority, reverse=True)
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

    # その他のメソッドは変更なし…
