# components/debug_panel.py

import streamlit as st


class DebugPanel:
    def render(self, llm_meta):
        st.subheader("🧠 LLM デバッグ")

        # ↓ GPT-4o / Hermes の本文比較は専用クラスに丸投げ
        self.model_viewer.render(llm_meta)
