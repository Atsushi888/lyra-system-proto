# llm/llm_manager.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

from llm.llm_router import LLMRouter


@dataclass
class LLMModelConfig:
    """
    1つの LLM モデルに関する設定情報。

    - name:       論理名（"gpt4o", "gpt51", "hermes" など）
    - vendor:     ベンダー種別（"openai", "openrouter", "xai", "google" など）
    - router_fn:  LLMRouter 内で呼び出すメソッド名（文字列）
    - priority:   優先度（大きいほど優先）
    - enabled:    有効フラグ
    - extra:      APIキー名や model_family などの追加情報
    """
    name: str
    vendor: str
    router_fn: str
    priority: float = 1.0
    enabled: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


class LLMManager:
    """
    LLM モデル群のメタ情報を管理するクラス。

    ・モデルの登録／優先度／有効／無効など「設定のレジストリ」が主な役割
    ・実際の API 呼び出しは、内部に持っている LLMRouter に委譲する
    """

    # persona_id ごとのシングルトン管理
    _POOL: Dict[str, "LLMManager"] = {}

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

        # Grok / Gemini はデフォルトでは「登録されているが disabled」
        # → UI から有効化させる前提
        manager.register_grok(priority=1.5, enabled=False)
        manager.register_gemini(priority=1.5, enabled=False)

        cls._POOL[persona_id] = manager
        return manager

    def __init__(self, persona_id: str = "default") -> None:
        # persona ごとなどに分けたい時用の識別子
        self.persona_id = persona_id
        # name -> LLMModelConfig
        self._models: Dict[str, LLMModelConfig] = {}
        # 実際の呼び出しを担当するルーター
        self._router = LLMRouter()

    # ==============================
    # モデル登録系
    # ==============================
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
        """汎用のモデル登録メソッド。"""
        cfg = LLMModelConfig(
            name=name,
            vendor=vendor,
            router_fn=router_fn,
            priority=priority,
            enabled=enabled,
            extra=extra or {},
        )
        self._models[name] = cfg

    # --- 便利メソッド（標準3モデル＋追加） --------------------

    def register_gpt4o(self, *, priority: float = 3.0, enabled: bool = True) -> None:
        """gpt-4o 系のモデル登録ヘルパ。"""
        self.register_model(
            "gpt4o",
            vendor="openai",
            router_fn="call_gpt4o",
            priority=priority,
            enabled=enabled,
            extra={
                "env_key": "OPENAI_API_KEY",
                "model_family": "gpt-4o",
            },
        )

    def register_gpt51(self, *, priority: float = 2.0, enabled: bool = True) -> None:
        """gpt-5.1 系のモデル登録ヘルパ。"""
        self.register_model(
            "gpt51",
            vendor="openai",
            router_fn="call_gpt51",
            priority=priority,
            enabled=enabled,
            extra={
                "env_key": "OPENAI_API_KEY",
                "model_family": "gpt-5.1",
            },
        )

    def register_hermes(self, *, priority: float = 1.0, enabled: bool = True) -> None:
        """Hermes 系（OpenRouter）のモデル登録ヘルパ。"""
        self.register_model(
            "hermes",
            vendor="openrouter",
            router_fn="call_hermes",
            priority=priority,
            enabled=enabled,
            extra={
                "env_key": "OPENROUTER_API_KEY",
                "model_family": "hermes",
            },
        )

    def register_grok(self, *, priority: float = 1.5, enabled: bool = True) -> None:
        """
        Grok（xAI）系モデルの登録ヘルパ。

        ※ 実際の呼び出しは LLMRouter.call_grok() を実装してそちらに委譲する。
        """
        self.register_model(
            "grok",
            vendor="xai",
            router_fn="call_grok",
            priority=priority,
            enabled=enabled,
            extra={
                # Streamlit secrets / env のどちらかに設定してもらう想定
                "env_key": "GROK_API_KEY",
                "model_family": "grok-2",
            },
        )

    def register_gemini(self, *, priority: float = 1.5, enabled: bool = True) -> None:
        """
        Gemini（Google）系モデルの登録ヘルパ。

        ※ 実際の呼び出しは LLMRouter.call_gemini() を実装してそちらに委譲する。
        """
        self.register_model(
            "gemini",
            vendor="google",
            router_fn="call_gemini",
            priority=priority,
            enabled=enabled,
            extra={
                "env_key": "GEMINI_API_KEY",
                "model_family": "gemini-2.0",
            },
        )

    # ==============================
    # 実際の呼び出し
    # ==============================
    def call_model(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        """
        ModelsAI や MemoryAI から呼ばれる実行用メソッド。

        - 登録されている model_name を探す
        - router_fn に対応する LLMRouter のメソッドを呼び出す
        - 返ってきた結果（text / (text, usage) など）をそのまま返す
        """
        cfg = self._models.get(model_name)
        if cfg is None:
            raise ValueError(f"Unknown model: {model_name}")

        fn = getattr(self._router, cfg.router_fn, None)
        if fn is None:
            raise AttributeError(
                f"LLMRouter has no method '{cfg.router_fn}' for model '{model_name}'"
            )

        # LLMRouter 側は `call_xxx(messages=..., **kwargs)` という想定
        return fn(messages=messages, **kwargs)

    # ==============================
    # 取得系
    # ==============================
    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        """ModelsAI / View で使いやすい形で返す。"""
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
        """priority の高い順にソートした dict を返す。"""
        items = sorted(
            self._models.items(),
            key=lambda kv: kv[1].priority,
            reverse=True,
        )
        result: Dict[str, Dict[str, Any]] = {}
        for name, cfg in items:
            result[name] = {
                "vendor": cfg.vendor,
                "router_fn": cfg.router_fn,
                "priority": cfg.priority,
                "enabled": cfg.enabled,
                "extra": dict(cfg.extra),
            }
        return result

    # ---- 利用可能モデル情報（Secrets/環境変数チェック込み） ----

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        env_key / secrets に API キーがあるかどうかも含めた props を返す。

        戻り値例:
        {
          "gpt4o": {
            "vendor": "openai",
            "router_fn": "call_gpt4o",
            "priority": 3.0,
            "enabled": True,  # 現在の設定上の enabled
            "extra": {...},
            "has_key": True,  # APIキーがあるかどうか
          },
          ...
        }
        """
        import os

        props = self.get_model_props()

        # Streamlit 無しの環境でも壊れないように遅延インポート＆例外保護
        try:
            import streamlit as st  # type: ignore
            secrets = st.secrets
        except Exception:  # pragma: no cover - Streamlit無い時用
            secrets = {}

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
        UI で選択された「使う/使わない」を反映する。
        """
        for name, cfg in self._models.items():
            if name in enabled:
                cfg.enabled = bool(enabled[name])

    # ==============================
    # YAML ロード（将来拡張用のダミー実装）
    # ==============================
    def load_default_config(self, path: Optional[str] = None) -> bool:
        """
        llm_default.yaml から設定を読み込む。

        無くても動くようにしてあるので、今は「読めれば True / 読めなければ False」くらいの扱い。
        """
        import os

        if path is None:
            path = "llm_default.yaml"

        if not os.path.exists(path):
            return False

        try:
            import yaml  # type: ignore[import]
        except Exception:
            # PyYAML が入っていなければ読み込みはスキップ
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

            self.register_model(
                name,
                vendor=vendor,
                router_fn=router_fn,
                priority=priority,
                enabled=enabled,
                extra=extra,
            )

        return True
