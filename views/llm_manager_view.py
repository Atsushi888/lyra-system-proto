# views/llm_manager_view.py
from __future__ import annotations

from typing import Dict, Any

import streamlit as st

# from llm.llm_manager_factory import get_llm_manager
from llm.llm_manager import LLMManager

class LLMManagerView:
    """
    ユーザー設定（LLM）画面用のビュー。

    - llm_default.yaml や環境変数から読んだ結果…という構想は残しつつ、
      まずは LLMManager に登録されているモデル一覧を確認する用途に特化。
    """

    TITLE = "LLM 設定 / 接続状況"

    def __init__(self) -> None:
        # グローバルな LLMManager を共有して利用
        self.manager = llmget_llm_manager()

    # ------------------------------------------------------------------
    def render(self) -> None:
        st.header("🧊 ユーザー設定（LLM）")
        st.subheader(self.TITLE)
        st.caption("llm_default.yaml と環境変数から読み取った LLM モデル一覧の確認ビューです。")

        props: Dict[str, Dict[str, Any]] = self.manager.get_model_props()

        if not props:
            st.info("現在、登録されている LLM モデルはありません。")
            return

        for name, cfg in props.items():
            with st.expander(f"モデル: {name}", expanded=True):
                enabled = cfg.get("enabled", False)
                vendor = cfg.get("vendor", "-")
                router_fn = cfg.get("router_fn", "-")
                priority = cfg.get("priority", 0.0)
                extra = cfg.get("extra", {})

                st.markdown(f"- ベンダー: `{vendor}`")
                st.markdown(f"- router_fn: `{router_fn}`")
                st.markdown(f"- priority: `{priority}`")
                st.markdown(f"- enabled: `{enabled}`")

                if extra:
                    st.markdown("**extra:**")
                    for k, v in extra.items():
                        st.markdown(f"  - `{k}`: `{v}`")


def create_llm_manager_view() -> LLMManagerView:
    """
    ModeSwitcher から呼び出すためのファクトリ関数。
    """
    return LLMManagerView()
