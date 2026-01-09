from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import os

try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    st = None  # type: ignore
    _HAS_ST = False

from llm.llm_ai.llm_adapters.base import BaseLLMAdapter


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
    # 内部：APIキー判定（env + streamlit.secrets）
    # ===========================================================
    @staticmethod
    def _has_api_key(env_key: Optional[str]) -> bool:
        if not env_key:
            return True
        if os.getenv(env_key):
            return True
        if _HAS_ST and isinstance(st.secrets, dict) and st.secrets.get(env_key):
            return True
        return False

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

        auto_vendor, auto_extra = self._infer_vendor_extra(name)

        # adapter 側に supported_parameters を持たせている場合にも拾えるようにする
        adapter_supported = getattr(adapter, "supported_parameters", None)
        if isinstance(adapter_supported, list) and adapter_supported:
            auto_extra = {**auto_extra, "supported_parameters": list(adapter_supported)}

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
        if model_name in ("gpt51", "gpt4o"):
            return "openai", {"env_key": "OPENAI_API_KEY", "model_family": model_name}
        if model_name == "gpt52":
            return "openai", {"env_key": "OPENAI_API_KEY", "model_family": "gpt52"}
        if model_name == "grok":
            return "xai", {"env_key": "GROK_API_KEY", "model_family": "grok"}
        if model_name == "gemini":
            return "google", {"env_key": "GEMINI_API_KEY", "model_family": "gemini"}
        if model_name in ("hermes", "hermes_new", "llama_unc"):
            return "openrouter", {"env_key": "OPENROUTER_API_KEY", "model_family": model_name}
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

        追加の安全策：
        - enabled=False のモデルは呼ばない
        - env_key が必要でキーが無いモデルは呼ばない
        """
        cfg = self._models.get(model_name)
        if cfg is None:
            raise ValueError(f"Unknown model: {model_name}")

        if not cfg.enabled:
            raise RuntimeError(f"Model disabled: {model_name}")

        env_key = (cfg.extra or {}).get("env_key")
        if env_key and not self._has_api_key(str(env_key)):
            raise RuntimeError(f"Missing API key: {env_key} (model={model_name})")

        call_params = dict(cfg.params)
        call_params.update(kwargs)

        return cfg.adapter.call(messages=messages, **call_params)

    # ===========================================================
    # 情報取得（互換）
    # ===========================================================
    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for name, cfg in self._models.items():
            extra = dict(cfg.extra)

            # ★ここが重要：ModelsAI2 側が参照できるようにトップレベルにも出す
            supported = extra.get("supported_parameters")
            if isinstance(supported, list):
                supported_parameters = list(supported)
            else:
                supported_parameters = []

            out[name] = {
                "vendor": cfg.vendor,
                "priority": cfg.priority,
                "enabled": cfg.enabled,
                "extra": extra,
                "params": dict(cfg.params),
                "defaults": dict(cfg.params),
                "supported_parameters": supported_parameters,
            }
        return out

    def get_models_sorted(self) -> Dict[str, Dict[str, Any]]:
        items = sorted(self._models.items(), key=lambda kv: kv[1].priority, reverse=True)
        out: Dict[str, Dict[str, Any]] = {}
        for name, cfg in items:
            extra = dict(cfg.extra)
            supported = extra.get("supported_parameters")
            if isinstance(supported, list):
                supported_parameters = list(supported)
            else:
                supported_parameters = []

            out[name] = {
                "vendor": cfg.vendor,
                "priority": cfg.priority,
                "enabled": cfg.enabled,
                "extra": extra,
                "params": dict(cfg.params),
                "defaults": dict(cfg.params),
                "supported_parameters": supported_parameters,
            }
        return out

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        “呼び出してよいモデル” だけを返す。
    
        条件:
        - enabled=True
        - env_key が必要な場合は has_key=True（env + secrets）
        """
        props = self.get_model_props()
    
        out: Dict[str, Dict[str, Any]] = {}
        for name, p in props.items():
            enabled = bool(p.get("enabled", True))
    
            extra = p.get("extra") or {}
            env_key = extra.get("env_key")
    
            has_key = self._has_api_key(str(env_key)) if env_key else True
    
            p2 = dict(p)
            p2["has_key"] = has_key
    
            if enabled and has_key:
                out[name] = p2
    
        return out


    def set_enabled_models(self, enabled: Dict[str, bool]) -> None:
        for name, flag in enabled.items():
            if name in self._models:
                self._models[name].enabled = bool(flag)
