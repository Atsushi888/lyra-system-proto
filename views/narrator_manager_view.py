# views/narrator_manager_view.py
from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from actors.narrator.narrator_manager import NarratorManager, NarratorCallLog


class NarratorManagerView:
    """
    NarratorManager の呼び出し履歴を可視化するビュー。

    ✅ 改善点
    - models_result の status / error / traceback / call_kwargs を表示
    - _meta / _system も表示（enabled_models等の確認に必須）
    - Judge candidates のキー不一致を修正（name / details 参照）
    """

    SESSION_KEY = "narrator_manager"

    def __init__(self) -> None:
        pass

    def _get_manager(self) -> NarratorManager:
        if self.SESSION_KEY not in st.session_state:
            st.session_state[self.SESSION_KEY] = NarratorManager(state=st.session_state)
        return st.session_state[self.SESSION_KEY]

    def render(self) -> None:
        self.render_main()

    def render_main(self) -> None:
        manager = self._get_manager()
        history = manager.get_history()
        last = manager.get_last()

        st.markdown("## 📝 Narrator Manager Debug View")
        st.caption("NarratorAI → LLM 呼び出しの履歴と、Models/Judge の結果を確認できます。")

        if not history:
            st.info("まだ Narrator の呼び出し履歴はありません。")
            return

        if last is not None:
            st.markdown("### 🔍 Latest Call")
            self._render_log_item(last, idx=1)

        st.markdown("### 📚 History (recent)")
        for i, log in enumerate(reversed(history[-10:]), start=1):
            if log is last:
                continue
            self._render_log_item(log, idx=i + 1)

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
                chosen = (log.judge_result or {}).get("chosen_model", "")
                st.write(f"chosen_model: `{chosen}`")
                st.markdown("---")

    # ----------------------------
    # 内部：モデル結果表示ヘルパ
    # ----------------------------
    @staticmethod
    def _as_dict(x: Any) -> Dict[str, Any]:
        return x if isinstance(x, dict) else {}

    def _render_models_result(self, models_result: Dict[str, Any]) -> None:
        if not isinstance(models_result, dict) or not models_result:
            st.caption("models_result is empty.")
            return

        # まず _meta / _system を上に出す（enabled_models確認用）
        meta = self._as_dict(models_result.get("_meta"))
        sys_ = self._as_dict(models_result.get("_system"))

        if meta:
            st.markdown("### _meta")
            st.json(meta)

        if sys_:
            st.markdown("### _system")
            st.json(sys_)

        st.markdown("### per-model results")

        # _meta/_system を除外して通常モデルだけ
        model_items = [(k, v) for k, v in models_result.items() if k not in ("_meta", "_system")]
        if not model_items:
            st.caption("No per-model entries.")
            return

        for model_name, info_any in model_items:
            info = self._as_dict(info_any)

            status = str(info.get("status") or "unknown")
            text = (info.get("text") or "").strip()
            error = info.get("error")
            tb = info.get("traceback")
            call_kwargs = info.get("call_kwargs") or {}

            # 見出し
            badge = "✅" if status == "ok" else "❌"
            st.markdown(f"#### {badge} {model_name}  (status=`{status}`)")

            # まず短い要約
            if text:
                st.markdown("**text (head):**")
                st.code(text[:400] + ("..." if len(text) > 400 else ""))

            # error（あれば常に出す）
            if error:
                st.markdown("**error:**")
                st.code(str(error))

            # call_kwargs（常に出す：爆死の原因特定に必須）
            if isinstance(call_kwargs, dict) and call_kwargs:
                with st.expander("call_kwargs (actually passed to LLM)", expanded=False):
                    st.json(call_kwargs)
            else:
                st.caption("call_kwargs: (empty)")

            # traceback（長いので折りたたみ）
            if tb:
                with st.expander("traceback", expanded=False):
                    st.code(str(tb))

            st.markdown("---")

    def _render_log_item(self, log: NarratorCallLog, idx: int) -> None:
        st.markdown(f"#### [{idx}] {log.label} ({log.task_type})")
        st.write(f"- mode: `{log.mode_current}`")

        with st.expander("📨 Prompt (messages)", expanded=False):
            for m in log.messages:
                role = m.get("role", "?")
                content = m.get("content", "")
                st.markdown(f"- **{role}**:")
                st.code(content)

        with st.expander("🤖 Models result (full)", expanded=True):
            self._render_models_result(log.models_result)

        with st.expander("⚖ Judge result", expanded=False):
            jr = log.judge_result or {}
            chosen = jr.get("chosen_model", "")
            st.write(f"chosen_model: `{chosen}`")
            chosen_text = (jr.get("chosen_text") or "").strip()
            if chosen_text:
                st.markdown("**chosen_text:**")
                st.markdown(chosen_text)

            # 候補（JudgeAI3 の candidates は name/details）
            candidates = jr.get("candidates") or []
            if candidates:
                st.markdown("**candidates:**")
                for c in candidates:
                    name = c.get("name", "?")
                    score = c.get("score", "?")
                    length = c.get("length", "?")
                    status = c.get("status", "?")
                    details = c.get("details") or {}
                    pr = details.get("priority_rank", None)

                    st.markdown(f"- `{name}` status={status} score={score} len={length}" + (f" prio_rank={pr}" if pr is not None else ""))
            reason = jr.get("reason")
            if reason:
                st.markdown("**reason:**")
                st.code(str(reason))

        with st.expander("🧾 Final text (used by NarratorAI)", expanded=True):
            st.markdown(log.final_text or "（空）")

        st.markdown("---")


def create_narrator_manager_view() -> NarratorManagerView:
    return NarratorManagerView()
