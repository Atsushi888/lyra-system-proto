# views/answertalker_view.py

from __future__ import annotations

from typing import Optional, Dict, Any, List

import streamlit as st

from actors.actor import Actor


def _auto_height(text: str, base: int = 120, per_line: int = 22, max_h: int = 420) -> int:
    """
    テキスト量に応じて text_area の高さを自動調整するための簡易関数。

    - ざっくり 40 文字 ≒ 1 行 とみなして行数を見積もる
    - base 以上、max_h 以下の範囲で高さを返す
    """
    if not text:
        return base
    approx_lines = max(1, len(text) // 40 + 1)
    h = approx_lines * per_line
    h = max(base, h)
    h = min(max_h, h)
    return h


class AnswerTalkerView:
    """
    AnswerTalker の動作確認用ビュー（閲覧専用）。

    - Actor インスタンスを受け取り、その中の AnswerTalker に紐づく llm_meta を“見るだけ”
    - 自分から AnswerTalker.run_models() や MemoryAI.update_from_turn() などは実行しない
    - llm_meta["models"] / ["judge"] / ["composer"] / ["memory_*"] を一覧表示する
    """

    def __init__(self, actor: Actor) -> None:
        self.actor = actor  # 今は未使用だが将来の拡張用に保持

    # ============================
    # models 表示
    # ============================
    def _render_models(self, llm_meta: Dict[str, Any]) -> None:
        st.markdown("### llm_meta に登録された AI 回答一覧（models）")

        models = llm_meta.get("models")
        if not isinstance(models, dict) or not models:
            st.info(
                "llm_meta['models'] に情報がありません。\n\n"
                "- AnswerTalker.run_models() がどこかでまだ実行されていないか、\n"
                "- もしくはモデルから有効な回答が返ってきていない可能性があります。"
            )
            return

        for model_name, info in models.items():
            st.markdown(f"#### モデル: `{model_name}`")

            if isinstance(info, dict):
                status = info.get("status", "unknown")
                text = info.get("text", "")
                err = info.get("error")

                st.write(f"- status: `{status}`")
                if err:
                    st.write(f"- error: `{err}`")

                if text:
                    st.write("**回答テキスト:**")
                    height = _auto_height(text)
                    st.text_area(
                        "text",
                        value=text,
                        height=height,
                        key=f"models_text_{model_name}",
                    )
                else:
                    st.write("（text フィールドがありません）")
            else:
                st.write("想定外の形式です:", info)

            st.markdown("---")

    # ============================
    # judge 表示
    # ============================
    def _render_judge(self, llm_meta: Dict[str, Any]) -> None:
        st.markdown("### JudgeAI2 の判定結果（llm_meta['judge']）")

        judge = llm_meta.get("judge")
        if not isinstance(judge, dict) or not judge:
            st.info(
                "llm_meta['judge'] に情報がありません。\n\n"
                "- JudgeAI2 がまだ実行されていないか、\n"
                "- あるいはエラーで判定に失敗している可能性があります。"
            )
            return

        status = judge.get("status", "unknown")
        chosen_model = judge.get("chosen_model", "")
        raw_reason = (
            judge.get("reason")
            or judge.get("reason_text")
            or judge.get("reason_detail")
        )
        error = judge.get("error")

        # reason を必ず文字列にしておく
        reason_str = ""
        if isinstance(raw_reason, (list, tuple)):
            reason_str = ", ".join(str(x) for x in raw_reason)
        elif raw_reason is not None:
            reason_str = str(raw_reason)

        chosen_text = judge.get("chosen_text", "")

        st.write(f"- status: `{status}`")
        if chosen_model:
            st.write(f"- chosen_model: `{chosen_model}`")
        if error:
            st.write(f"- error: `{error}`")

        # ★ 選択理由（必ず 1 行テキストとして表示）
        if reason_str.strip():
            st.write("**選択理由（reason）:**")
            st.text(reason_str)

            comment = judge.get("reason_text")
            if comment:
                st.write("**Judge コメント（reason_text）:**")
                st.text(comment)

            # breakdown 表示（カンマ区切り前提）
            parts = [p.strip() for p in reason_str.split(",") if p.strip()]
            if parts:
                st.write("- breakdown:")
                for p in parts:
                    st.markdown(f"    - **{p}**")

        # 採用テキスト
        if chosen_text:
            st.write("**採用テキスト（chosen_text）:**")
            h = _auto_height(chosen_text)
            st.text_area(
                "chosen_text",
                value=chosen_text,
                height=h,
                key="judge_chosen_text",
            )

        # 候補一覧（スコアなど）
        candidates: Any = judge.get("candidates")
        st.markdown("#### 候補モデル一覧（candidates）")
        if isinstance(candidates, list) and candidates:
            for i, c in enumerate(candidates):
                if not isinstance(c, dict):
                    continue
                model = c.get("model", "")
                st.markdown(f"- 候補 {i+1}: `{model}`")
                st.write(
                    f"  - status: `{c.get('status', 'unknown')}`  "
                    f"/ score: `{c.get('score', '')}`  "
                    f"/ length: `{c.get('length', '')}`"
                )
                if c.get("error"):
                    st.write(f"  - error: `{c.get('error')}`")
                details = c.get("details")
                if isinstance(details, List) and details:
                    st.write("  - details:")
                    for d in details:
                        st.write(f"    - {d}")
                st.write("")
        else:
            st.write("（candidates がありません）")

        st.markdown("---")

    # ============================
    # composer 表示
    # ============================
    def _render_composer(self, llm_meta: Dict[str, Any]) -> None:
        st.markdown("### ComposerAI の最終結果（llm_meta['composer']）")

        comp = llm_meta.get("composer")
        if not isinstance(comp, dict) or not comp:
            st.info(
                "llm_meta['composer'] に情報がありません。\n\n"
                "- ComposerAI がまだ実行されていないか、\n"
                "- あるいはエラーで compose に失敗している可能性があります。"
            )
            return

        status = comp.get("status", "unknown")
        text = comp.get("text", "")
        error = comp.get("error")
        source_model = comp.get("source_model")
        mode = comp.get("mode")
        refiner_meta = comp.get("refiner") or {}

        st.write(f"- status: `{status}`")
        if source_model:
            st.write(f"- source_model: `{source_model}`")
        if mode:
            st.write(f"- mode: `{mode}`")
        if error:
            st.write(f"- error: `{error}`")

        if refiner_meta:
            st.write(
                f"- refiner: model=`{refiner_meta.get('model', 'N/A')}`, "
                f"used=`{refiner_meta.get('used', False)}`, "
                f"status=`{refiner_meta.get('status', '-')}`, "
                f"error=`{refiner_meta.get('error', '')}`"
            )

        if text:
            st.write("**最終返答テキスト（composer.text）:**")
            h = _auto_height(text)
            st.text_area(
                "composer_text",
                value=text,
                height=h,
                key="composer_text_area",
            )

        # オプション：summary があれば表示
        summary = comp.get("summary")
        if summary:
            st.write("**サマリ（summary）:**")
            h2 = _auto_height(summary, base=80)
            st.text_area(
                "composer_summary",
                value=summary,
                height=h2,
                key="composer_summary_area",
            )

        st.markdown("---")

    # ============================
    # memory 表示（MemoryAI v0.1 / JSON 永続化版）
    # ============================
    def _render_memory(self, llm_meta: Dict[str, Any]) -> None:
        st.markdown("### MemoryAI の状態（長期記憶）")

        # Actor → AnswerTalker → MemoryAI をたどる
        memory_ai = getattr(getattr(self.actor, "answer_talker", None), "memory_ai", None)

        if memory_ai is None:
            st.info(
                "MemoryAI インスタンスが取得できませんでした。\n\n"
                "- Actor 内に answer_talker.memory_ai が存在しないか、\n"
                "- まだ MemoryAI を組み込んでいない可能性があります。"
            )
            return

        # 属性は存在チェックしつつ表示（古い実装との互換も考慮）
        persona_id = getattr(memory_ai, "persona_id", "(unknown)")
        max_items = getattr(memory_ai, "max_store_items", None)
        file_path = getattr(memory_ai, "file_path", None)

        st.write(f"- persona_id: `{persona_id}`")
        if max_items is not None:
            st.write(f"- max_store_items: `{max_items}`")
        if file_path:
            st.write(f"- storage_file: `{file_path}`")
        else:
            st.write("- storage_file: (unknown)")

        # ---- 保存済み記憶一覧 ----
        try:
            records = memory_ai.get_all_records()
        except Exception as e:
            records = []
            st.warning(f"MemoryRecord の取得に失敗しました: {e}")

        st.markdown("#### 保存済み MemoryRecord 一覧")

        if not records:
            st.info("現在、保存済みの MemoryRecord はありません。")
        else:
            for i, rec in enumerate(records, start=1):
                try:
                    # dataclass の場合と dict の場合の両方に対応
                    summary = getattr(rec, "summary", None) or rec.get("summary", "")
                    importance = getattr(rec, "importance", None) or rec.get("importance", "")
                    created_at = getattr(rec, "created_at", None) or rec.get("created_at", "")
                    rid = getattr(rec, "id", None) or rec.get("id", "")
                except Exception:
                    continue

                st.text_input(
                    f"記憶 {i}: [imp={importance}] {summary[:40]}",
                    value=summary,
                    key=f"memory_record_{i}",
                )
                st.caption(f"id={rid} / created_at={created_at}")

        st.markdown("---")

        # ---- llm_meta 内の memory_* 情報 ----
        st.markdown("### llm_meta 内のメモリ関連メタ情報")

        mem_ctx = llm_meta.get("memory_context")
        mem_update = llm_meta.get("memory_update")

        st.write(f"- memory_context: `{mem_ctx if mem_ctx is not None else '（未設定）'}`")

        if not isinstance(mem_update, dict):
            st.write("- memory_update: （未設定）")
            return

        st.write("**memory_update（直近ターンの記憶更新結果）:**")
        status = mem_update.get("status", "unknown")
        added = mem_update.get("added", 0)
        total = mem_update.get("total", 0)
        reason = mem_update.get("reason", "")
        error = mem_update.get("error")

        st.write(f"- status: `{status}`")
        st.write(f"- added: `{added}`")
        st.write(f"- total: `{total}`")
        if reason:
            st.write(f"- reason: {reason}")
        if error:
            st.write(f"- error: {error}")

        # 追加されたレコードのダンプ（あれば）
        recs = mem_update.get("records")
        if isinstance(recs, list) and recs:
            st.write("#### added records（このターンで追加された記憶）")
            for i, r in enumerate(recs, start=1):
                if not isinstance(r, dict):
                    continue
                s = r.get("summary", "")
                imp = r.get("importance", "")
                st.text_input(
                    f"added {i}: [imp={imp}] {s[:40]}",
                    value=s,
                    key=f"memory_update_added_{i}",
                )

        # LLM 生テキスト（raw_reply）があればデバッグ用に表示
        raw_reply = mem_update.get("raw_reply")
        if raw_reply:
            st.write("#### raw_reply（MemoryAI 抽出用 LLM の生テキスト）:")
            h = _auto_height(str(raw_reply), base=80)
            st.text_area(
                "memory_raw_reply",
                value=str(raw_reply),
                height=h,
                key="memory_raw_reply_area",
            )

        st.markdown("---")

    # ============================
    # メイン render
    # ============================
    def render(self) -> None:
        st.title("AnswerTalker / ModelsAI・JudgeAI2・ComposerAI・MemoryAI デバッグビュー（閲覧専用）")

        st.markdown(
            "この画面では、Actor に紐づく AnswerTalker が保持している "
            "`llm_meta` の内容（models / judge / composer / memory）を参照できます。"
        )
        st.markdown(
            "> ※ この画面からは AnswerTalker.run_models() や MemoryAI.update_from_turn() などは実行しません。"
            " 会談システムや別の処理でパイプラインを走らせたあとに開いてください。"
        )

        st.markdown("---")

        llm_meta: Optional[Dict[str, Any]] = st.session_state.get("llm_meta")
        if not isinstance(llm_meta, dict):
            st.info(
                "llm_meta がまだ初期化されていません。\n\n"
                "- 会談システムや AnswerTalker を利用する処理を一度実行してから、\n"
                "- 再度この画面を開いてください。"
            )
            return

        self._render_models(llm_meta)
        self._render_judge(llm_meta)
        self._render_composer(llm_meta)
        self._render_memory(llm_meta)


# ModeSwitcher などから呼びやすくするラッパ
def render(actor: Actor) -> None:
    view = AnswerTalkerView(actor)
    view.render()
