from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st


class LLMManager:
    """
    LLM モデル群の登録・参照を一括管理するクラス。

    - モデル定義（vendor / router_fn / priority / enabled / extra）
    - get_model_props() で View / Actor から参照
    - get_or_create() で Streamlit の session_state を介して
      「persona ごとに 1 個」の LLMManager を共有する
    """

    # session_state 内で使うキー
    SESSION_KEY = "llm_managers"

    def __init__(self, persona_id: str = "default") -> None:
        self.persona_id = persona_id
        # name -> props
        self._models: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # クラスメソッド: persona ごとのインスタンスを取得
    # ------------------------------------------------------------------
    @classmethod
    def get_or_create(cls, persona_id: str = "default") -> "LLMManager":
        """
        persona_id ごとに 1 個の LLMManager を session_state 内に確保して返す。
        まだ存在しない場合は新規作成し、デフォルトモデルを登録する。
        """
        registry = st.session_state.get(cls.SESSION_KEY)
        if not isinstance(registry, dict):
            registry = {}

        manager = registry.get(persona_id)
        if not isinstance(manager, cls):
            # 新規作成
            manager = cls(persona_id=persona_id)
            manager._register_builtin_defaults()
            registry[persona_id] = manager

        st.session_state[cls.SESSION_KEY] = registry
        return manager

    # ------------------------------------------------------------------
    # モデル登録系
    # ------------------------------------------------------------------
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
        """
        汎用のモデル登録関数。
        """
        self._models[name] = {
            "vendor": vendor,
            "router_fn": router_fn,
            "priority": float(priority),
            "enabled": bool(enabled),
            "extra": extra or {},
        }

    def register_gpt4o(self, priority: float = 3.0, enabled: bool = True) -> None:
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

    def register_gpt51(self, priority: float = 2.0, enabled: bool = True) -> None:
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

    def register_hermes(self, priority: float = 1.0, enabled: bool = True) -> None:
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

    def _register_builtin_defaults(self) -> None:
        """
        新規作成時に呼ばれるデフォルト登録処理。
        既に何か登録されていたら何もしない。
        """
        if self._models:
            return

        self.register_gpt4o(priority=3.0, enabled=True)
        self.register_gpt51(priority=2.0, enabled=True)
        self.register_hermes(priority=1.0, enabled=True)

    # ------------------------------------------------------------------
    # 参照系
    # ------------------------------------------------------------------
    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        """
        現在登録されているモデル定義を返す。
        （変更防止のため浅いコピーを返す）
        """
        return dict(self._models)

    def get_model(self, name: str) -> Optional[Dict[str, Any]]:
        return self._models.get(name)
