from __future__ import annotations

from typing import Any, Dict, Optional, Callable, List
import os

from llm.router_openai import call_gpt4o, call_gpt51
from llm.router_openrouter import call_hermes


class LLMManager:
    """
    - LLM モデル管理クラス
    - register_xxx() / get_model_props() を提供
    - call_model() を全モデル共通の呼び出しインターフェースとして提供
    """

    def __init__(self, persona_id: str = "default") -> None:
        self.persona_id = persona_id

        # モデル定義の辞書
        #   models = {
        #       "gpt4o": {
        #           "vendor": "openai",
        #           "router_fn": call_gpt4o,
        #           "priority": 3.0,
        #           "enabled": True,
        #           "extra": {
        #               "env_key": "OPENAI_API_KEY",
        #               "model_family": "gpt4o",
        #           }
        #       },
        #       ...
        #   }
        self.models: Dict[str, Dict[str, Any]] = {}

    # -------------------------------------------------------
    # ■ モデル登録メソッド
    # -------------------------------------------------------

    def register_gpt4o(self, priority: float = 3.0, enabled: bool = True) -> None:
        self.models["gpt4o"] = {
            "vendor": "openai",
            "router_fn": call_gpt4o,
            "priority": priority,
            "enabled": enabled,
            "extra": {
                "env_key": "OPENAI_API_KEY",
                "model_family": "gpt4o",
            },
        }

    def register_gpt51(self, priority: float = 2.0, enabled: bool = True) -> None:
        self.models["gpt51"] = {
            "vendor": "openai",
            "router_fn": call_gpt51,
            "priority": priority,
            "enabled": enabled,
            "extra": {
                "env_key": "OPENAI_API_KEY",
                "model_family": "gpt-5.1",
            },
        }

    def register_hermes(self, priority: float = 1.0, enabled: bool = True) -> None:
        self.models["hermes"] = {
            "vendor": "openrouter",
            "router_fn": call_hermes,
            "priority": priority,
            "enabled": enabled,
            "extra": {
                "env_key": "OPENROUTER_API_KEY",
                "model_family": "hermes",
            },
        }

    # -------------------------------------------------------
    # ■ モデル構成まとめ取得（AnswerTalker → JudgeAI が使用）
    # -------------------------------------------------------

    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        """
        AnswerTalker → JudgeAI2 などが参照するメタ情報を返す。
        """
        return self.models

    # -------------------------------------------------------
    # ■ モデル呼び出し（ModelsAI から使用される）
    # -------------------------------------------------------

    def call_model(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        各モデルに共通の呼び出し口。

        ModelsAI.collect() から呼ばれる。
        """
        if model_name not in self.models:
            return {
                "status": "error",
                "error": f"model '{model_name}' not registered",
                "text": "",
            }

        cfg = self.models[model_name]

        # disabled はスキップ
        if not cfg.get("enabled", False):
            return {
                "status": "disabled",
                "text": "",
            }

        router_fn: Callable = cfg["router_fn"]

        # API キー確認（vendorごとに）
        env_key = cfg["extra"].get("env_key")
        api_key = os.environ.get(env_key) if env_key else None

        if env_key and not api_key:
            return {
                "status": "error",
                "error": f"missing API key: {env_key}",
                "text": "",
            }

        # 実際の LLM 呼び出し
        try:
            text = router_fn(messages=messages, api_key=api_key)
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "text": "",
            }

        return {
            "status": "ok",
            "text": text or "",
        }
