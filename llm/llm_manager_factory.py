# llm/llm_manager_factory.py
from __future__ import annotations

from typing import Optional

import streamlit as st

from .llm_manager import LLMManager

# アプリ全体で共有する LLMManager を session_state に 1 個だけ持つ
_SESSION_KEY = "llm_manager_global"


def _create_manager() -> LLMManager:
    """
    内部用: 新しい LLMManager を生成し、デフォルトモデルを登録する。
    """
    manager = LLMManager()
    manager.register_default_models()
    return manager


def get_llm_manager(persona_id: Optional[str] = None) -> LLMManager:
    """
    アプリ全体で共有する LLMManager を返すファクトリ関数。

    以前は persona_id ごとに別インスタンスを持っていたが、
    現在は「グローバル 1 個」を共有する方針に変更している。

    引数 persona_id は後方互換のために残しているが、
    現在の実装では無視される。
    """
    manager = st.session_state.get(_SESSION_KEY)

    if not isinstance(manager, LLMManager):
        manager = _create_manager()
        st.session_state[_SESSION_KEY] = manager

    return manager
