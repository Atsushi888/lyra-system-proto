# components/model_viewer.py

from typing import Any, Dict
import streamlit as st


class MultiModelViewer:
    """
    llm_meta に入っている各モデルの reply を縦に並べて表示するビューア。
    """

    def __init__(self, title: str = "モデル比較"):
        self.title = title

        # 将来モデルを増やしたときはここに追記するだけでOK
        self.model_labels = {
            "gpt4o": "GPT-4o",
            "hermes": "Hermes",
            # "claude": "Claude 3" みたいに増やせる
        }

    def render(self, llm_meta: Dict[str, Any] | None) -> None:
        if not llm_meta:
            st.caption("（まだレスポンスがありません）")
            return

        st.markdown(f"### {self.title}")

        has_any = False
        for key, label in self.model_labels.items():
            info = llm_meta.get(key)
            if not info:
                continue

            has_any = True
            reply = info.get("reply", "（返信なし）")

            st.markdown(f"#### {label}")
            st.write(reply)
            st.markdown("---")

        if not has_any:
            st.caption("（表示可能なモデルがありません）")
