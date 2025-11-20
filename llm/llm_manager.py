# llm/llm_manager.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import streamlit as st


@dataclass
class LLMModelConfig:
    name: str                 # 内部キー（"gpt4o" など）
    router_fn: str            # LLMRouter 上のメソッド名
    label: str                # UI 用表示名
    priority: float = 1.0     # JudgeAI2 用の重み
    vendor: str = "custom"    # "openai" / "groq" / ...
    enabled: bool = True
    required_env: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # None を消しておくと扱いやすい
        if d.get("required_env") is None:
            d["required_env"] = []
        return d


class LLMManager:
    """
    persona_id ごとに LLM の設定を管理するクラス。

    - get_model_props(): 画面描画や JudgeAI2 に渡す設定を返す
    - register_gpt4o / gpt51 / hermes(): よく使うモデルの登録ヘルパ
    - get(persona_id): persona ごとのシングルトンを返すクラスメソッド
    """

    # ★ レジストリを session_state に置くためのキー
    _SESSION_REG_KEY = "llm_manager_registry_v2"

    # -------- クラスメソッド: レジストリ管理 --------

    @classmethod
    def _get_registry(cls) -> Dict[str, "LLMManager"]:
        reg = st.session_state.get(cls._SESSION_REG_KEY)
        if not isinstance(reg, dict):
            reg = {}
            st.session_state[cls._SESSION_REG_KEY] = reg
        return reg

    @classmethod
    def get(cls, persona_id: str = "default") -> "LLMManager":
        """
        persona_id ごとに 1 個だけ LLMManager インスタンスを返す。

        まだ無ければ新規作成し、デフォルトモデルを登録する。
        """
        reg = cls._get_registry()
        mgr = reg.get(persona_id)

        if mgr is None:
            mgr = cls(persona_id=persona_id)

            # ここで「初期状態ならデフォルトモデルを登録」
            if not mgr.get_model_props():
                mgr.register_gpt4o(priority=3.0, enabled=True)
                mgr.register_gpt51(priority=2.0, enabled=True)
                mgr.register_hermes(priority=1.0, enabled=True)

            reg[persona_id] = mgr
            st.session_state[cls._SESSION_REG_KEY] = reg

        return mgr

    # -------- インスタンス部分 --------

    def __init__(self, persona_id: str = "default") -> None:
        self.persona_id = persona_id
        # name -> props dict
        self._models: Dict[str, Dict[str, Any]] = {}

    # モデル登録系 -----------------------

    def register_model(self, cfg: LLMModelConfig) -> None:
        d = cfg.to_dict()
        name = d["name"]
        self._models[name] = d

    # ショートカット: gpt4o / gpt51 / hermes

    def register_gpt4o(self, priority: float = 3.0, enabled: bool = True) -> None:
        self.register_model(
            LLMModelConfig(
                name="gpt4o",
                router_fn="call_gpt4o",
                label="GPT-4o",
                priority=priority,
                vendor="openai",
                enabled=enabled,
                required_env=["OPENAI_API_KEY"],
            )
        )

    def register_gpt51(self, priority: float = 2.0, enabled: bool = True) -> None:
        self.register_model(
            LLMModelConfig(
                name="gpt51",
                router_fn="call_gpt51",
                label="GPT-5.1",
                priority=priority,
                vendor="openai",
                enabled=enabled,
                required_env=["OPENAI_API_KEY"],
            )
        )

    def register_hermes(self, priority: float = 1.0, enabled: bool = True) -> None:
        self.register_model(
            LLMModelConfig(
                name="hermes",
                router_fn="call_hermes",
                label="Hermes",
                priority=priority,
                vendor="groq",
                enabled=enabled,
                required_env=["GROQ_API_KEY"],
            )
        )

    # 参照系 ---------------------------

    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        """
        name -> props の dict を返す。
        AnswerTalker / ModelsAI / LLMManagerView から参照される。
        """
        return dict(self._models)

    def as_list(self) -> List[Dict[str, Any]]:
        """UI 用に list でも欲しくなった時用のヘルパ。"""
        return list(self._models.values())
