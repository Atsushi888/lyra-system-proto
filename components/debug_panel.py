# components/debug_panel.py

from __future__ import annotations

from typing import Any, Dict
import json

import streamlit as st

from deliberation.multi_ai_response import MultiAIResponse


class DebugPanel:
    """
    LLM呼び出しのメタ情報と、
    マルチAI審議の結果（Judge）をサイドバーに表示するパネル。
    """

    def __init__(self, title: str = "Debug Panel") -> None:
        self.title = title
        self.multi_ai = MultiAIResponse()

        # ここに追加していくだけでモデルを増やせる
        # key: llm_meta["models"] のキー名 / value: 表示ラベル
        self.model_labels: Dict[str, str] = {
            "gpt4o": "GPT-4o",
            "hermes": "Hermes",
            # "claude": "Claude 3" など増やしてOK
        }

    # ===== 内部ヘルパ：モデル比較ビュー（旧 MultiModelViewer の復活版） =====
    def _render_model_compare(self, models: Dict[str, Any]) -> None:
        st.markdown("#### 🤖 モデル比較")

        has_any = False
        for key, label in self.model_labels.items():
            info = models.get(key)
            if not isinstance(info, dict):
                continue

            has_any = True
            reply = info.get("reply") or info.get("text") or "（返信なし）"

            st.markdown(f"**{label}**")
            st.write(reply)

            usage = info.get("usage") or info.get("usage_main")
            if isinstance(usage, dict) and usage:
                pt = usage.get("prompt_tokens", "？")
                ct = usage.get("completion_tokens", "？")
                tt = usage.get("total_tokens", "？")
                st.caption(
                    f"tokens: total={tt}, prompt={pt}, completion={ct}"
                )

            st.markdown("---")

        if not has_any:
            st.caption("（表示可能なモデルがありません）")

    # ===== 内部ヘルパ：Judge結果表示 =====
    def _render_multi_ai_result(self, judge: Dict[str, Any] | None) -> None:
        st.markdown("#### ⚖️ Multi AI Judge")

        if not isinstance(judge, dict):
            st.caption("（審議結果はまだありません）")
            return

        winner = judge.get("winner", "？")
        score = judge.get("score_diff", 0.0)
        comment = judge.get("comment", "")

        cols = st.columns(2)
        cols[0].metric("勝者", winner)
        cols[1].metric(
            "スコア差",
            f"{score:.2f}" if isinstance(score, (int, float)) else score,
        )

        if comment:
            st.markdown("**理由:**")
            st.write(comment)

        with st.expander("🪶 JudgeAI raw", expanded=False):
            raw = judge.get("raw")
            if raw:
                st.code(str(raw), language="text")
            pair = judge.get("pair")
            if pair:
                st.caption(f"比較ペア: {pair}")

    # ===== 公開：描画本体 =====
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

        # --- プロンプトプレビュー ---
        prompt_preview = llm_meta.get("prompt_preview")
        if isinstance(prompt_preview, str) and prompt_preview.strip():
            with st.expander("📝 プロンプトプレビュー", expanded=False):
                st.code(prompt_preview, language="text")

        # --- models を使った GPT-4o / Hermes 比較 ---
        models = llm_meta.get("models")
        if isinstance(models, dict) and models:
            with st.expander("🤝 モデル応答比較", expanded=True):
                self._render_model_compare(models)

        # --- MultiAIResponse による Judge 結果を表示 ---
        #     （ここで JudgeAI を呼ぶのではなく、multi_ai_response に任せる）
        agg = self.multi_ai.process(llm_meta)
        judge = agg.get("judge")

        with st.expander("⚖️ マルチAI審議結果", expanded=True):
            self._render_multi_ai_result(judge)

        # --- 生 llm_meta を最後に置いておく ---
        with st.expander("raw llm_meta (開発者向け)", expanded=False):
            try:
                st.code(
                    json.dumps(llm_meta, ensure_ascii=False, indent=2),
                    language="json",
                )
            except Exception:
                st.code(str(llm_meta), language="text")
