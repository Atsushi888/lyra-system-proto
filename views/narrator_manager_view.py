# views/narrator_manager_view.py
from __future__ import annotations

from typing import Any

import streamlit as st

from actors.narrator.narrator_manager import NarratorManager, NarratorCallLog


class NarratorManagerView:
    """
    NarratorManager の呼び出し履歴を可視化するビュー。

    - ModeSwitcher から「デバッグモード」としてメイン画面に表示
    - 必要なら他画面からサイドバー表示も可能（render_sidebar）
    """

    SESSION_KEY = "narrator_manager"

    def __init__(self) -> None:
        pass

    def _get_manager(self) -> NarratorManager:
        if self.SESSION_KEY not in st.session_state:
            # state に session_state を渡すことで、履歴がセッションに残る
            st.session_state[self.SESSION_KEY] = NarratorManager(state=st.session_state)
        return st.session_state[self.SESSION_KEY]

    # ===== メイン画面用：ModeSwitcher から呼ぶ =====
    def render(self) -> None:
        """ModeSwitcher 互換の render(). メインビューとして使う。"""
        self.render_main()

    def render_main(self) -> None:
        manager = self._get_manager()
        history = manager.get_history()
        last = manager.get_last()

        st.markdown("## 📝 Narrator Manager Debug View")
        st.caption("NarratorAI → LLM 呼び出しの履歴と、Judge の選択結果を確認できます。")

        if not history:
            st.info("まだ Narrator の呼び出し履歴はありません。")
            return

        # 直近の結果を上に、その下に履歴一覧
        if last is not None:
            st.markdown("### 🔍 Latest Call")
            self._render_log_item(last, idx=1)

        st.markdown("### 📚 History (recent)")
        # 直近 10 件くらいを表示（必要なら数は調整）
        for i, log in enumerate(reversed(history[-10:]), start=1):
            if log is last:
                continue
            self._render_log_item(log, idx=i + 1)

    # ===== サイドバー用：Council などから添え物として見る場合 =====
    def render_sidebar(self) -> None:
        manager = self._get_manager()
        history = manager.get_history()

        with st.sidebar.expander("📝 Narrator Manager Log", expanded=False):
            if not history:
                st.caption("（まだ Narrator の呼び出し履歴はありません）")
                return

            for idx, log in enumerate(reversed(history[-5:]), start=1):
                st.markdown(f"**[{idx}] {log.label} ({log.task_type})**")
                st.write(f"mode: `{log.mode_current}`")
                chosen = log.judge_result.get("chosen_model", "")
                st.write(f"chosen_model: `{chosen}`")
                st.markdown("---")

    # ===== 内部：1件分の詳細描画 =====
    def _render_log_item(self, log: NarratorCallLog, idx: int) -> None:
        st.markdown(f"#### [{idx}] {log.label} ({log.task_type})")
        st.write(f"- mode: `{log.mode_current}`")

        with st.expander("📨 Prompt (messages)", expanded=False):
            for m in log.messages:
                role = m.get("role", "?")
                content = m.get("content", "")
                st.markdown(f"- **{role}**:")
                st.code(content)

        with st.expander("🤖 Models result (summary)", expanded=False):
            for model_name, info in log.models_result.items():
                text = (info.get("text") or "").strip()
                st.markdown(f"- **{model_name}**")
                if text:
                    st.markdown(
                        f"    - text: {text[:200]}{'...' if len(text) > 200 else ''}"
                    )

        with st.expander("⚖ Judge result", expanded=False):
            chosen = log.judge_result.get("chosen_model", "")
            st.write(f"chosen_model: `{chosen}`")
            chosen_text = (log.judge_result.get("chosen_text") or "").strip()
            if chosen_text:
                st.markdown("**chosen_text:**")
                st.markdown(chosen_text)

            # ★ 追加：候補モデルとスコア・理由を一覧表示
            candidates = log.judge_result.get("candidates") or []
            if candidates:
                st.markdown("**candidates:**")
                for c in candidates:
                    m = c.get("model", "?")
                    score = c.get("score", "?")
                    reason = c.get("reason", "")
                    st.markdown(f"- `{m}` (score={score})")
                    if reason:
                        st.markdown(f"    - {reason}")

        with st.expander("🧾 Final text (used by NarratorAI)", expanded=True):
            st.markdown(log.final_text or "（空）")

        st.markdown("---")


def create_narrator_manager_view() -> NarratorManagerView:
    """ModeSwitcher 用のファクトリ."""
    return NarratorManagerView()
