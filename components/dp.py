from __future__ import annotations

from typing import Any, Dict
import json
import streamlit as st

# from deliberation.multi_ai_response import MultiAIResponse


class DebugPanel:
    """
    LLM呼び出しのメタ情報と、
    マルチAIレスポンスシステムをまとめて表示するパネル。

    ・基本情報（route, model_main, tokens など）
    ・raw llm_meta
    ・マルチAI関連（MultiAIResponse に丸投げ）
    """

    def __init__(self, title: str = "Debug Panel") -> None:
        self.title = title
        # self.multi_ai_response = MultiAIResponse()

    def render(self, llm_meta: Dict[str, Any] | None) -> None:
        st.markdown(f"### 🛠 {self.title}")

        if not isinstance(llm_meta, dict) or not llm_meta:
            st.caption("（まだメタ情報がありません）")
            return

        # --- 基本情報 ---
        route = llm_meta.get("route")
        model_main = llm_meta.get("model_main")
        usage_main = llm_meta.get("usage_main") or llm_meta.get("usage")

        with st.expander("基本情報", expanded=False):
            if route:
                st.write(f"- route: `{route}`")
            if model_main:
                st.write(f"- model_main: `{model_main}`")
            if isinstance(usage_main, dict):
                pt = usage_main.get("prompt_tokens", "？")
                ct = usage_main.get("completion_tokens", "？")
                tt = usage_main.get("total_tokens", "？")
                st.write(f"- tokens: total={tt}, prompt={pt}, completion={ct}")

        # --- マルチAIレスポンス（表示も審議も全部ここに委譲） ---
        with st.expander("🧪 マルチAIレスポンスシステム", expanded=True):
            # self.multi_ai_response.render(llm_meta)

        # --- raw llm_meta ---
        with st.expander("raw llm_meta (開発者向け)", expanded=False):
            try:
                st.code(
                    json.dumps(llm_meta, ensure_ascii=False, indent=2),
                    language="json",
                )
            except Exception:
                st.code(str(llm_meta), language="text")
