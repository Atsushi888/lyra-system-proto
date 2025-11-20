# views/llm_manager_view.py

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from llm.llm_manager import LLMManager


class LLMManagerView:
    """
    LLMManager の設定・状態をざっと確認するためのビュー。

    - LLMManager.get_model_props() の中身を一覧表示するだけのシンプル版。
    - ここから先、編集UIや llm_default.yaml のロード結果表示などを
      どんどん拡張していく想定。
    """

    def __init__(self, manager: Optional[LLMManager] = None) -> None:
        # とりあえずデフォルトの LLMManager を 1 個だけ作る
        self.manager = manager or LLMManager()

    def render(self) -> None:
        st.markdown("## LLM 設定 / 接続状況")
        st.caption(
            "llm_default.yaml と環境変数から読み取った LLM モデル一覧の確認ビューです。"
        )

        model_props: Dict[str, Dict[str, Any]] = self.manager.get_model_props()

        if not model_props:
            st.info("現在、登録されている LLM モデルはありません。")
            return

        st.markdown("### 登録済み LLM モデル一覧")

        for name, props in model_props.items():
            label = props.get("label", name)
            enabled = props.get("enabled", True)
            available = props.get("available", True)
            vendor = props.get("vendor", "-")
            priority = props.get("priority", 0.0)
            required_env = props.get("required_env", [])

            header = f"{label} ({name})"
            if not enabled:
                header += " — [disabled]"
            elif not available:
                header += " — [env NG]"

            with st.expander(header, expanded=False):
                cols = st.columns(3)
                cols[0].metric("enabled", "✅" if enabled else "❌")
                cols[1].metric("available", "✅" if available else "❌")
                cols[2].metric("priority", f"{priority:.1f}")

                st.write(f"**vendor:** {vendor}")
                if required_env:
                    st.write("**required env:** " + ", ".join(required_env))
                else:
                    st.write("**required env:** (なし)")

                st.markdown("**raw props:**")
                st.json(props, expanded=False)


def create_llm_manager_view() -> LLMManagerView:
    """
    ModeSwitcher から呼ばれるファクトリ関数。

    components.mode_switcher では
        from views.llm_manager_view import create_llm_manager_view
    と import して、routes の 'USER' などに
        "view": create_llm_manager_view
    の形で渡せば OK。
    """
    return LLMManagerView()
