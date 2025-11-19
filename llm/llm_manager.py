# llm/llm_manager.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Iterable, Optional
import os


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

    - register_model / unregister_model
    - enable / disable
    - list_models / get_model_props
    - 必要な環境変数が足りないモデルは available=False として扱う
    """

    def __init__(self) -> None:
        self._models: Dict[str, LLMModelConfig] = {}

    # -----------------------------
    # 登録・削除
    # -----------------------------
    def register_model(self, cfg: LLMModelConfig) -> None:
        self._models[cfg.name] = cfg

    def unregister_model(self, name: str) -> None:
        self._models.pop(name, None)

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

        戻り値例:
            [
              {
                "name": "gpt4o",
                "label": "GPT-4o",
                "enabled": True,
                "priority": 3.0,
                "vendor": "openai",
                "router_fn": "call_gpt4o",
                "available": True,
                "missing_env": [],
              },
              ...
            ]
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
    # 既存コードとの互換用: model_props 生成
    # -----------------------------
    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        """
        AnswerTalker / ModelsAI / JudgeAI2 が期待している
        model_props 形式に変換して返す。

        {
          "gpt4o": {
            "enabled": True,
            "priority": 3.0,
            "router_fn": "call_gpt4o",
            "label": "GPT-4o",
            "vendor": "openai",
          },
          ...
        }
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
        """
        enabled かつ available なモデルのみを返す。
        """
        base = self.get_model_props()
        return {
            name: p
            for name, p in base.items()
            if p.get("enabled")
        }
