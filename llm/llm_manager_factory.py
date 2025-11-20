# llm/llm_manager_factory.py

from __future__ import annotations

from typing import Optional

import streamlit as st

from llm.llm_manager import LLMManager

# session_state に格納するキー
_LLM_MANAGER_KEY = "llm_manager"


def get_llm_manager(persona_id: str = "default") -> LLMManager:
    """
    Streamlit の session_state を使って、
    セッション内でただ 1 つの LLMManager インスタンスを返すファクトリ。

    - 既に作られていればそれを返す
    - 無ければ llm_default.yaml からロードして作成し、session_state に保存
    - YAML が無い / PyYAML 無しなどの場合は、ハードコード登録でフォールバック
    """
    # すでに存在すればそれを使う
    mgr = st.session_state.get(_LLM_MANAGER_KEY)
    if isinstance(mgr, LLMManager):
        # 必要ならここで後から persona_id を上書きしてもよい
        # mgr.persona_id = persona_id
        return mgr

    # まだ無い場合は新規に構築
    try:
        mgr = LLMManager.from_yaml(
            path="config/llm_default.yaml",
            persona_id=persona_id,
        )
        st.session_state[_LLM_MANAGER_KEY] = mgr
        return mgr
    except FileNotFoundError:
        print(
            "[LLMManagerFactory] config/llm_default.yaml が見つからないため、"
            "ハードコードされたデフォルト設定を使用します。"
        )
    except RuntimeError as e:
        # PyYAML 未導入など
        print(
            f"[LLMManagerFactory] YAML 読み込みエラーのため、"
            f"ハードコードされたデフォルト設定を使用します: {e}"
        )

    # フォールバック：ハードコード版
    mgr = LLMManager(persona_id=persona_id)
    mgr.register_gpt4o(priority=3.0, enabled=True)
    mgr.register_gpt51(priority=4.0, enabled=True)
    mgr.register_hermes(priority=2.0, enabled=True)

    st.session_state[_LLM_MANAGER_KEY] = mgr
    return mgr
