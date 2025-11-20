# llm/llm_manager_factory.py

from __future__ import annotations

from typing import Dict
import streamlit as st

from .llm_manager import LLMManager

# persona_id → LLMManager を突っ込んでおくストア
_SESSION_KEY = "llm_managers"


def _get_store() -> Dict[str, LLMManager]:
    store = st.session_state.get(_SESSION_KEY)
    if not isinstance(store, dict):
        store = {}
        st.session_state[_SESSION_KEY] = store
    return store


def get_llm_manager(persona_id: str = "default") -> LLMManager:
    """
    persona_id ごとに 1 個だけ LLMManager を作って共有するファクトリ。

    - まだ Manager が無い場合は生成して、デフォルトLLMを register_* する
    - 以降は同じインスタンスを返す
    """
    store = _get_store()

    mgr = store.get(persona_id)
    if mgr is None:
        mgr = LLMManager(persona_id=persona_id)

        # ここでデフォルトLLMを登録する（旧 _build_default_llm_manager の中身）
        if not mgr.get_model_props():  # 念のため空のときだけ
            mgr.register_gpt4o(priority=3.0, enabled=True)
            mgr.register_gpt51(priority=2.0, enabled=True)
            mgr.register_hermes(priority=1.0, enabled=True)

        store[persona_id] = mgr
        st.session_state[_SESSION_KEY] = store

    return mgr
