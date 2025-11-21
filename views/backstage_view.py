# views/backstage_view.py
from __future__ import annotations
import streamlit as st
import os
import json

from components.debug_panel import DebugPanel


class BackstageView:
    def render(self) -> None:
        llm_meta = st.session_state.get("llm_meta")
        DebugPanel(title="Lyra Backstage – Multi AI Debug View").render(llm_meta)

        st.markdown("---")
        st.subheader("MemoryAI デバッグ")

        if st.button("floria_ja のメモリファイルを診断する"):
            path = "data/memory/floria_ja.json"
            st.write(f"対象ファイル: `{path}`")

            if not os.path.exists(path):
                st.error("ファイルが存在しません。まだ一度も記憶が保存されていない可能性があります。")
            else:
                st.success("ファイルは存在します。")

                size = os.path.getsize(path)
                st.write(f"- ファイルサイズ: `{size}` バイト")

                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as e:
                    st.error(f"JSON の読み込みに失敗しました: {e}")
                    return

                if isinstance(data, list):
                    st.write(f"- JSON はリストです。要素数: `{len(data)}`")
                    if data:
                        st.write("- 先頭3件のプレビュー:")
                        st.json(data[:3])
                    else:
                        st.info("リストは空です（記憶が 0 件です）。")
                else:
                    st.write(f"- JSON の型: `{type(data)}`")
                    st.json(data)
