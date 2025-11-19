# llm/llm_manager.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Iterable, Optional
import os

from llm.llm_router import LLMRouter


@dataclass
class LLMModelConfig:
    """
    個々の LLM モデルに関する設定情報。
    - name         : 内部名（例: "gpt4o", "gpt51", "hermes"）
    - router_fn    : LLMRouter 上のメソッド名（例: "call_gpt4o"）
    - label        : 表示用ラベル
    - priority     : JudgeAI2 での優先度
    - enabled      : 有効/無効フラグ
    - vendor       : "openai", "openrouter" などのベンダ名（UI 用）
    - required_env : 利用に必要な環境変数名の一覧
    """

    name: str
    router_fn: str
    label: str
    priority: float = 1.0
    enabled: bool = True
    vendor: str = "generic"
    required_env: List[str] = field(default_factory=list)

    def is_available(self) -> bool:
        """
        必要な環境変数が揃っているかの簡易チェック。
        1つでも欠けていれば False を返す。
        """
        for key in self.required_env:
            if not os.getenv(key):
                return False
        return True


class LLMManager:
    """
    利用可能な LLM の一覧・増減・状態を管理するクラス。

    役割:
    - モデル定義のレジストリ
    - 有効/無効管理
    - APIキーの有無チェック
    - model_props の生成（既存コード互換）
    - 名前指定で LLM を呼び出す窓口（call_model）

    実際の HTTP 呼び出しなどは内部の LLMRouter に委譲する。
    """

    def __init__(self, router: LLMRouter | None = None) -> None:
        self._models: Dict[str, LLMModelConfig] = {}
        self.router: LLMRouter = router or LLMRouter()

    # -----------------------------
    # 基本: 汎用登録・削除
    # -----------------------------
    def register_model(self, cfg: LLMModelConfig) -> None:
        self._models[cfg.name] = cfg

    def unregister_model(self, name: str) -> None:
        self._models.pop(name, None)

    # -----------------------------
    # 便利メソッド: 代表的な LLM を登録
    # -----------------------------
    def register_model_gpt4o(
        self,
        priority: float = 3.0,
        enabled: bool = True,
    ) -> None:
        cfg = LLMModelConfig(
            name="gpt4o",
            router_fn="call_gpt4o",
            label="GPT-4o",
            priority=priority,
            enabled=enabled,
            vendor="openai",
            required_env=["OPENAI_API_KEY"],
        )
        self.register_model(cfg)

    def register_model_gpt51(
        self,
        priority: float = 2.0,
        enabled: bool = True,
    ) -> None:
        cfg = LLMModelConfig(
            name="gpt51",
            router_fn="call_gpt51",
            label="GPT-5.1",
            priority=priority,
            enabled=enabled,
            vendor="openai",
            required_env=["OPENAI_API_KEY"],
        )
        self.register_model(cfg)

    def register_model_hermes4(
        self,
        priority: float = 1.0,
        enabled: bool = True,
    ) -> None:
        cfg = LLMModelConfig(
            name="hermes",
            router_fn="call_hermes",
            label="Hermes 4",
            priority=priority,
            enabled=enabled,
            vendor="openrouter",
            required_env=["OPENROUTER_API_KEY"],
        )
        self.register_model(cfg)

    def register_model_grok41(
        self,
        priority: float = 2.0,
        enabled: bool = True,
    ) -> None:
        cfg = LLMModelConfig(
            name="grok41",
            router_fn="call_grok41",
            label="Grok 4.1",
            priority=priority,
            enabled=enabled,
            vendor="xai",
            required_env=["XAI_API_KEY"],
        )
        self.register_model(cfg)

    # -----------------------------
    # 有効・無効切り替え
    # -----------------------------
    def enable(self, name: str) -> None:
        cfg = self._models.get(name)
        if cfg is not None:
            cfg.enabled = True

    def disable(self, name: str) -> None:
        cfg = self._models.get(name)
        if cfg is not None:
            cfg.enabled = False

    # -----------------------------
    # 参照
    # -----------------------------
    def get(self, name: str) -> Optional[LLMModelConfig]:
        return self._models.get(name)

    def iter_models(self) -> Iterable[LLMModelConfig]:
        return self._models.values()

    def list_models(self) -> List[Dict[str, Any]]:
        """
        UI 用。現在登録されている LLM 一覧と状態を返す。
        """
        items: List[Dict[str, Any]] = []
        for cfg in self._models.values():
            missing_env = [
                k for k in cfg.required_env if not os.getenv(k)
            ]
            items.append(
                {
                    "name": cfg.name,
                    "label": cfg.label,
                    "enabled": cfg.enabled,
                    "priority": cfg.priority,
                    "vendor": cfg.vendor,
                    "router_fn": cfg.router_fn,
                    "available": cfg.is_available(),
                    "missing_env": missing_env,
                }
            )
        return items

    # -----------------------------
    # model_props 生成（既存コード互換）
    # -----------------------------
    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        """
        AnswerTalker / ModelsAI / JudgeAI2 が期待している
        model_props 形式に変換して返す。
        """
        props: Dict[str, Dict[str, Any]] = {}
        for cfg in self._models.values():
            props[cfg.name] = {
                "enabled": cfg.enabled and cfg.is_available(),
                "priority": cfg.priority,
                "router_fn": cfg.router_fn,
                "label": cfg.label,
                "vendor": cfg.vendor,
            }
        return props

    def get_enabled_model_props(self) -> Dict[str, Dict[str, Any]]:
        base = self.get_model_props()
        return {
            name: p
            for name, p in base.items()
            if p.get("enabled")
        }

    # -----------------------------
    # モデル呼び出し窓口
    # -----------------------------
    def call_model(
        self,
        name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> Any:
        """
        name で指定されたモデルを呼び出す統一インターフェース。

        - LLMModelConfig.router_fn を参照して LLMRouter 上のメソッドを解決
        - 見つからない / 無効 / 環境変数不足の場合は RuntimeError を投げる
        """
        cfg = self._models.get(name)
        if cfg is None:
            raise RuntimeError(f"unknown model: {name}")

        if not cfg.enabled or not cfg.is_available():
            raise RuntimeError(f"model '{name}' is disabled or unavailable")

        fn_name = cfg.router_fn
        fn = getattr(self.router, fn_name, None)
        if fn is None:
            raise RuntimeError(f"router has no method '{fn_name}'")

        # LLMRouter 側のインターフェースに合わせて呼び出す
        return fn(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
