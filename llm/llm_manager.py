# llm/llm_manager.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class LLMModelConfig:
    """
    1 モデル分の設定情報。

    name:
        モデル識別名（"gpt4o", "gpt51", "hermes" など）

    vendor:
        ベンダー識別用の文字列（"openai", "openrouter" など）

    router_fn:
        LLMRouter 上の呼び出し関数名（"call_gpt4o" など）

    priority:
        モデルの優先度。JudgeAI2 等での重みづけに使う想定。

    enabled:
        利用するかどうかのフラグ。

    extra:
        env_key など、拡張用の任意のメタ情報。
    """

    name: str
    vendor: str
    router_fn: str
    priority: float = 1.0
    enabled: bool = True
    extra: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # extra が None のときは空 dict にしておく
        if data.get("extra") is None:
            data["extra"] = {}
        return data


class LLMManager:
    """
    LLM モデル群の設定・管理クラス。

    役割:
      - 利用可能な LLM モデルのメタ情報を一元管理する
      - アプリ全体で共有されることを前提（factory 経由で取得）

    想定される利用箇所:
      - AnswerTalker / ModelsAI / JudgeAI2 などのバックエンド
      - LLM 設定ビュー（ユーザー設定画面）
    """

    def __init__(self) -> None:
        # name -> config_dict
        self._models: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # 共通インターフェース
    # ------------------------------------------------------------------
    def register_model(
        self,
        name: str,
        *,
        vendor: str,
        router_fn: str,
        priority: float = 1.0,
        enabled: bool = True,
        **extra: Any,
    ) -> None:
        """
        任意の LLM モデルを登録するための汎用メソッド。
        """
        cfg = LLMModelConfig(
            name=name,
            vendor=vendor,
            router_fn=router_fn,
            priority=priority,
            enabled=enabled,
            extra=extra or {},
        )
        self._models[name] = cfg.to_dict()

    def unregister_model(self, name: str) -> None:
        """
        モデル設定を削除する（存在しなければ黙って無視）。
        """
        self._models.pop(name, None)

    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        """
        現在登録されている全モデル設定を返す。

        戻り値は shallow copy にしておくことで、
        呼び出し側から直接内部 dict をいじれないようにしている。
        """
        return dict(self._models)

    def get_model(self, name: str) -> Optional[Dict[str, Any]]:
        """
        単一モデルの設定を取得。無ければ None。
        """
        return self._models.get(name)

    # ------------------------------------------------------------------
    # プリセットモデル登録
    # ------------------------------------------------------------------
    def register_gpt4o(self, *, priority: float = 3.0, enabled: bool = True) -> None:
        """
        OpenAI GPT-4o 用のプリセット。
        """
        self.register_model(
            "gpt4o",
            vendor="openai",
            router_fn="call_gpt4o",
            priority=priority,
            enabled=enabled,
            env_key="OPENAI_API_KEY",
            model_family="gpt-4o",
        )

    def register_gpt51(self, *, priority: float = 2.0, enabled: bool = True) -> None:
        """
        GPT-5.1 (仮) 用プリセット。
        """
        self.register_model(
            "gpt51",
            vendor="openai",
            router_fn="call_gpt51",
            priority=priority,
            enabled=enabled,
            env_key="OPENAI_API_KEY",
            model_family="gpt-5.1",
        )

    def register_hermes(self, *, priority: float = 1.0, enabled: bool = True) -> None:
        """
        Hermes (OpenRouter 経由など) 用プリセット。
        """
        self.register_model(
            "hermes",
            vendor="openrouter",
            router_fn="call_hermes",
            priority=priority,
            enabled=enabled,
            env_key="OPENROUTER_API_KEY",
            model_family="hermes",
        )

    def register_default_models(self) -> None:
        """
        デフォルトの 3 モデルを登録するユーティリティ。

        - gpt4o
        - gpt51
        - hermes
        """
        self.register_gpt4o(priority=3.0, enabled=True)
        self.register_gpt51(priority=2.0, enabled=True)
        self.register_hermes(priority=1.0, enabled=True)
