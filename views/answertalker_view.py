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
    - 自分から AnswerTalker.run_models() を呼んだりはしない
    - llm_meta["models"] / ["judge"] / ["composer"] / ["memory_*"] を一覧表示する
    - さらに、可能であれば AnswerTalker.memory_ai.memories の中身も一覧表示する
    """

    def __init__(self, actor: Actor) -> None:
        self.actor = actor  # AnswerTalker / MemoryAI への入口として保持

    # ============================
    # 内部: MemoryAI 取得ヘルパ
    # ============================
    def _get_memory_ai(self) -> Any:
        """
        Actor -> AnswerTalker -> MemoryAI を辿って memory_ai を取得する。

        - actor.answer_talker.memory_ai
        - または actor.talker.memory_ai
        のような構造を想定しつつ、防御的にたどる。
        見つからなければ None を返す。
        """
        actor = self.actor
        if actor is None:
            return None

        # よくありそうな名前を順に試す
        talker = getattr(actor, "answer_talker", None)
        if talker is None:
            talker = getattr(actor, "talker", None)

        if talker is None:
            return None

        mem_ai = getattr(talker, "memory_ai", None)
        return mem_ai

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
            # ここは一行で見えた方が分かりやすいので text を使う
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
                model_name = c.get("name") or c.get("model") or ""
                st.markdown(f"- 候補 {i+1}: `{model_name}`")
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
                elif isinstance(details, str) and details.strip():
                    # カンマ区切りなら行分割
                    parts = [p.strip() for p in details.split(",") if p.strip()]
                    if parts:
                        st.write("  - details:")
                        for p in parts:
                            st.write(f"    - {p}")
                    else:
                        st.write(f"  - details: {details}")
                elif details is not None:
                    st.write(f"  - details: {details}")
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
        source_model = comp.get("source_model", "")
        mode = comp.get("mode", "")
        summary = comp.get("summary", "")
        composer_info = comp.get("composer")
        refiner_info = comp.get("refiner")

        st.write(f"- status: `{status}`")
        if source_model:
            st.write(f"- source_model: `{source_model}`")
        if mode:
            st.write(f"- mode: `{mode}`")
        if composer_info and isinstance(composer_info, dict):
            name = composer_info.get("name")
            ver = composer_info.get("version")
            if name or ver:
                st.write(f"- composer: `{name or ''}` (version {ver or '-'})")
        if refiner_info and isinstance(refiner_info, dict):
            r_model = refiner_info.get("model") or "N/A"
            r_used = refiner_info.get("used")
            r_status = refiner_info.get("status")
            r_err = refiner_info.get("error")
            st.write(
                f"- refiner: model=`{r_model}`, used=`{r_used}`, status=`{r_status}`, error=`{r_err or ''}`"
            )
        if error:
            st.write(f"- error: `{error}`")

        if summary:
            st.write("**サマリ（summary）:**")
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
    # memory 表示（MemoryRecord ビュー）
    # ============================
    def _render_memory(self, llm_meta: Dict[str, Any]) -> None:
        st.markdown("### MemoryAI の状態（長期記憶）")

        mem_ai = self._get_memory_ai()
        if mem_ai is None:
            st.info(
                "MemoryAI インスタンスが取得できませんでした。\n\n"
                "- Actor 内に answer_talker.memory_ai が存在しないか、\n"
                "- まだ MemoryAI を組み込んでいない可能性があります。"
            )
        else:
            # persona_id / file_path などあれば軽く表示
            persona_id = getattr(mem_ai, "persona_id", "unknown")
            file_path = getattr(mem_ai, "file_path", "(unknown)")
            max_items = getattr(mem_ai, "max_store_items", None)

            st.write(f"- persona_id: `{persona_id}`")
            st.write(f"- storage_file: `{file_path}`")
            if max_items is not None:
                st.write(f"- max_store_items: `{max_items}`")

            memories = getattr(mem_ai, "memories", [])
            if isinstance(memories, list) and memories:
                st.markdown(f"#### 保存済み MemoryRecord 一覧（{len(memories)} 件）")

                # importance / created_at でソート（MemoryAI.build_memory_context と揃える）
                try:
                    sorted_mems = sorted(
                        memories,
                        key=lambda m: (getattr(m, "importance", 1), getattr(m, "created_at", "")),
                        reverse=True,
                    )
                except Exception:
                    sorted_mems = memories

                for idx, mem in enumerate(sorted_mems):
                    try:
                        mid = getattr(mem, "id", "")
                        importance = getattr(mem, "importance", 1)
                        created_at = getattr(mem, "created_at", "")
                        summary = getattr(mem, "summary", "")
                        tags = getattr(mem, "tags", [])
                        round_id = getattr(mem, "round_id", None)
                        src_user = getattr(mem, "source_user", "")
                        src_assistant = getattr(mem, "source_assistant", "")
                    except Exception:
                        st.write(f"- 記憶 {idx+1}: 取得中にエラーが発生しました。")
                        continue

                    with st.expander(f"記憶 {idx+1}: [imp={importance}] {summary[:40]}"):
                        st.write(f"- id: `{mid}`")
                        if round_id is not None:
                            st.write(f"- round_id: `{round_id}`")
                        st.write(f"- importance: `{importance}`")
                        st.write(f"- created_at: `{created_at}`")
                        if tags:
                            st.write(f"- tags: `{', '.join(tags)}`")

                        if summary:
                            st.write("**summary:**")
                            h = _auto_height(summary, base=60)
                            st.text_area(
                                f"mem_summary_{idx}",
                                value=summary,
                                height=h,
                            )

                        if src_user:
                            st.write("**source_user:**")
                            h = _auto_height(src_user, base=60)
                            st.text_area(
                                f"mem_src_user_{idx}",
                                value=src_user,
                                height=h,
                            )

                        if src_assistant:
                            st.write("**source_assistant:**")
                            h = _auto_height(src_assistant, base=60)
                            st.text_area(
                                f"mem_src_assistant_{idx}",
                                value=src_assistant,
                                height=h,
                            )
            else:
                st.info("現在、保存済みの MemoryRecord はありません。")

        st.markdown("---")

        # llm_meta に積んでいるメモリ関連の情報も表示
        st.markdown("#### llm_meta 内のメモリ関連メタ情報")

        mem_ctx = llm_meta.get("memory_context")
        mem_update = llm_meta.get("memory_update")
        mem_error = llm_meta.get("memory_update_error")

        if mem_ctx:
            st.write("**memory_context（次ターン用に挿入されたテキスト）:**")
            h = _auto_height(mem_ctx, base=80)
            st.text_area(
                "memory_context",
                value=mem_ctx,
                height=h,
                key="memory_context_area",
            )
        else:
            st.write("- memory_context: （未設定）")

        if isinstance(mem_update, dict) and mem_update:
            st.write("**memory_update（直近ターンの記憶更新結果）:**")
            added = mem_update.get("added")
            reason = mem_update.get("reason")
            raw_reply = mem_update.get("raw_reply")

            if reason:
                st.write(f"- reason: `{reason}`")

            if isinstance(added, list) and added:
                st.write(f"- added: {len(added)} 件")
                for i, rec in enumerate(added):
                    if not isinstance(rec, dict):
                        continue
                    st.write(f"  - added[{i}]:")
                    for k, v in rec.items():
                        st.write(f"    - {k}: {v}")
            else:
                st.write("- added: []")

            if raw_reply:
                st.write("**raw_reply（MemoryAI 抽出用 LLM の生テキスト）:**")
                h = _auto_height(raw_reply, base=80)
                st.text_area(
                    "memory_raw_reply",
                    value=raw_reply,
                    height=h,
                    key="memory_raw_reply_area",
                )
        else:
            st.write("- memory_update: （未設定）")

        if mem_error:
            st.write(f"- memory_update_error: `{mem_error}`")

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
