from typing import Any, Dict, Optional

import streamlit as st


class MultiAIJudgeResultView:
    """
    JudgeAI が出した審議結果を表示するだけのビュークラス。

    期待する judge dict の例:
        {
            "winner": "gpt4o",
            "score_diff": 0.7,
            "comment": "…理由…",
            "raw_text": "... LLMの生テキスト ...",
            "raw_json": { "winner": "A", "score_diff": 0.7, "comment": "…" },
            "route": "gpt",
            "pair": {"A": "gpt4o", "B": "hermes"},
        }
    """

    def __init__(self, title: str = "Multi AI Judge") -> None:
        self.title = title

    def render(self, judge: Optional[Dict[str, Any]]) -> None:
        st.subheader(self.title)

        # まだ審議結果がない場合
        if not isinstance(judge, dict):
            st.caption("（審議結果はまだありません）")
            return

        winner = judge.get("winner") or "―"
        score_diff = judge.get("score_diff", 0.0)
        comment = judge.get("comment") or ""

        # 勝者・スコア差
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**勝者**")
            st.write(winner)
        with cols[1]:
            st.markdown("**スコア差**")
            try:
                st.write(f"{float(score_diff):.2f}")
            except Exception:
                st.write(score_diff)

        # 理由
        st.markdown("**理由:**")
        if comment:
            st.write(comment)
        else:
            st.caption("（理由テキストなし）")

        raw_json = judge.get("raw_json")
        raw_text = judge.get("raw_text")
        pair = judge.get("pair")

        # 生データ表示（デバッグ用）
        with st.expander("🪵 JudgeAI raw", expanded=False):
            if isinstance(raw_json, dict):
                st.caption("parsed JSON")
                st.json(raw_json)

            if isinstance(raw_text, str) and raw_text.strip():
                st.caption("original text")
                st.code(raw_text, language="json")

            if isinstance(pair, dict):
                st.caption("比較ペア")
                st.write(pair)
