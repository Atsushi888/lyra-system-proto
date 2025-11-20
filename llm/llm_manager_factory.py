# llm/llm_manager_factory.py

from __future__ import annotations

import os
from typing import Dict

import streamlit as st

from llm.llm_manager import LLMManager

# persona_id ごとに LLMManager を保持する session_state のキー
SESSION_KEY = "LLM_MANAGER_BY_PERSONA"

# デフォルト設定ファイル（必要に応じてパスは調整）
DEFAULT_CONFIG_PATH = "config/llm_default.yaml"


def _load_default_models(manager: LLMManager) -> None:
    """
    初期化直後の LLMManager に対して、llm_default.yaml などから
    モデル定義を読み込むヘルパ。

    - すでに models が入っていれば何もしない
    - LLMManager 側に用意されているメソッド名の違いを吸収する
    """
    # すでに何か登録されていれば二重ロードはしない
    if getattr(manager, "models", None):
        return

    # 1) 専用メソッドがあればそれを優先
    if hasattr(manager, "load_default_models"):
        manager.load_default_models()
        return

    if hasattr(manager, "load_from_defaults"):
        manager.load_from_defaults()
        return

    # 2) フォールバック: YAML から直接ロード
    if hasattr(manager, "load_from_yaml"):
        path = DEFAULT_CONFIG_PATH
        if os.path.exists(path):
            manager.load_from_yaml(path)
        else:
            # ファイルが無いときは静かにスキップ（必要なら warning）
            st.warning(f"llm_default.yaml が見つかりませんでした: {path}")
        return

    # 3) どのメソッドも無ければ何もしない（将来の互換性のため）
    return


def get_llm_manager(persona_id: str = "default") -> LLMManager:
    """
    persona_id ごとに 1 つの LLMManager インスタンスを返すファクトリ。

    - 初回生成時に _load_default_models() を呼び出し、
      llm_default.yaml などからモデル定義を読み込む。
    - 以降は session_state から同じインスタンスを再利用する。
    """
    store: Dict[str, LLMManager] | None = st.session_state.get(SESSION_KEY)  # type: ignore[assignment]
    if not isinstance(store, dict):
        store = {}

    manager = store.get(persona_id)
    if manager is None:
        # 初回生成
        manager = LLMManager(persona_id=persona_id)

        # ★ ここでデフォルトモデルをロード（A案）
        _load_default_models(manager)

        store[persona_id] = manager
        st.session_state[SESSION_KEY] = store

    return manager
