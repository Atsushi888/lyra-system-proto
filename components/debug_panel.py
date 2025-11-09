# components/debug_panel.py

import streamlit as st


class DebugPanel:
    def render(self, llm_meta):
        st.subheader("🧠 LLM デバッグ")

        if not llm_meta:
            st.write("（まだレスポンスがありません）")
            return

        gpt4o = llm_meta.get("gpt4o")
        hermes = llm_meta.get("hermes")
        if gpt4o or hermes:
            # ↓ GPT-4o / Hermes の本文比較は専用クラスに丸投げ
            self.model_viewer.render(llm_meta)
        else:
            st.write("比較用データがありません。")
