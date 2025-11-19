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
                # JudgeAI2 側のキーは name
                model = c.get("name", "")
                st.markdown(f"- 候補 {i+1}: `{model}`")
                st.write(
                    f"  - status: `{c.get('status', 'unknown')}`  "
                    f"/ score: `{c.get('score', '')}`  "
                    f"/ length: `{c.get('length', '')}`"
                )
                if c.get("error"):
                    st.write(f"  - error: `{c.get('error')}`")
                details = c.get("details")
                if isinstance(details, list) and details:
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

        # refiner 情報
        if isinstance(refiner_meta, dict) and refiner_meta:
            st.write(
                "- refiner: "
                f"model=`{refiner_meta.get('model', 'N/A')}`, "
                f"used=`{refiner_meta.get('used', False)}`, "
                f"status=`{refiner_meta.get('status', '')}`, "
                f"error=`{refiner_meta.get('error', '')}`"
            )

        # サマリ
        summary = comp.get("summary")
        if summary:
            st.markdown("**サマリ（summary）:**")
            h = _auto_height(summary, base=80)
            st.text_area(
                "composer_summary",
                value=summary,
                height=h,
                key="composer_summary_area",
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

        st.markdown("---")

    # ============================
    # memory 表示（MemoryAI v2 対応）
    # ============================
    def _render_memory(self, llm_meta: Dict[str, Any]) -> None:
        st.markdown("### MemoryAI の状態（長期記憶）")

        # Actor → AnswerTalker → MemoryAI をたどる
        answer_talker = getattr(self.actor, "answer_talker", None)
        memory_ai = getattr(answer_talker, "memory_ai", None) if answer_talker else None

        if memory_ai is None:
            st.info(
                "MemoryAI インスタンスが取得できませんでした。\n\n"
                "- Actor 内に answer_talker.memory_ai が存在しないか、\n"
                "- まだ MemoryAI を組み込んでいない可能性があります。"
            )
        else:
            persona_id = getattr(memory_ai, "persona_id", "(unknown)")
            max_records = getattr(memory_ai, "max_records", None)

            st.write(f"- persona_id: `{persona_id}`")
            if max_records is not None:
                st.write(f"- max_records: `{max_records}`")
            st.write("- storage: session_state（インメモリ）")

            # 記憶一覧の取得（MemoryAI v2: get_all_records）
            try:
                records = memory_ai.get_all_records()
            except Exception as e:
                records = []
                st.warning(f"MemoryRecord の取得に失敗しました: {e}")

            if not records:
                st.info("現在、保存済みの MemoryRecord はありません。")
            else:
                st.markdown("#### 保存済み MemoryRecord 一覧")
                for rec in records:
                    label = f"記憶: [id={rec.id}, round={rec.round_id}]"
                    st.text_input(
                        label,
                        value=rec.summary,
                        key=f"memory_rec_{rec.id}",
                    )

        st.markdown("---")
        st.markdown("### llm_meta 内のメモリ関連メタ情報")

        # memory_context
        mem_ctx = llm_meta.get("memory_context")
        if mem_ctx:
            h = _auto_height(str(mem_ctx))
            st.text_area(
                "memory_context",
                value=str(mem_ctx),
                height=h,
                key="memory_context_area",
            )
        else:
            st.write("- memory_context: （未設定）")

        # memory_update: {status, added, total, error}
        mem_update = llm_meta.get("memory_update")
        if isinstance(mem_update, dict) and mem_update:
            st.write("- memory_update:")
            st.write(f"  - status: `{mem_update.get('status', '')}`")
            st.write(f"  - added: `{mem_update.get('added', '')}`")
            st.write(f"  - total: `{mem_update.get('total', '')}`")
            err = mem_update.get("error")
            if err:
                st.write(f"  - error: `{err}`")
        else:
            st.write("- memory_update: （未設定）")

        extra_err = llm_meta.get("memory_update_error")
        if extra_err:
            st.write(f"- memory_update_error: `{extra_err}`")

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
            "> ※ この画面からは AnswerTalker.run_models() や "
            "MemoryAI.update_from_turn() などは実行しません。"
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
