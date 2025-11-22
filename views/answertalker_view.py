# views/answertalker_view.py

from __future__ import annotations

from typing import Any, Dict, List, Protocol

import os
import json
import streamlit as st

from auth.roles import Role  # いまは未使用だが将来の拡張用に残しておく
from actors.actor import Actor
from actors.answer_talker import AnswerTalker
from personas.persona_floria_ja import Persona  # いまはフローリア固定


class View(Protocol):
    def render(self) -> None:
        ...


class AnswerTalkerView:
    """
    AnswerTalker / ModelsAI / JudgeAI2 / ComposerAI / MemoryAI の
    デバッグ・閲覧用ビュー。
    """

    TITLE = "🧩 AnswerTalker（AI統合テスト）"

    def __init__(self) -> None:
        # Actor と AnswerTalker を初期化
        # （必要なら将来ここに persona 選択 UI を付けられる）
        persona = Persona()
        self.actor = Actor("floria", persona)
        self.answer_talker = AnswerTalker(persona)

    def render(self) -> None:
        st.header(self.TITLE)

        st.info(
            "この画面では、Actor に紐づく AnswerTalker が保持している llm_meta の内容 "
            "（models / judge / composer / memory）を参照できます。\n\n"
            "※ この画面からは AnswerTalker.run_models() や MemoryAI.update_from_turn() などは実行しません。"
        )

        llm_meta: Dict[str, Any] = st.session_state.get("llm_meta", {}) or {}

        # ---- models ----
        st.subheader("llm_meta に登録された AI 回答一覧（models）")
        models = llm_meta.get("models", {})

        if not models:
            st.info("models 情報はまだありません。")
        else:
            for name, info in models.items():
                with st.expander(f"モデル: {name}", expanded=True):
                    status = info.get("status", "unknown")
                    text = info.get("text", "") or ""
                    usage = info.get("usage")
                    error = info.get("error")

                    st.write("- status:", status)
                    st.write("- len(text):", len(text))

                    if usage is not None:
                        st.write("- usage:", usage)

                    if error:
                        st.error(f"error: {error}")

                    # 実際のテキストも頭だけ確認できるように
                    if text:
                        st.markdown("**preview:**")
                        st.code(text[:1000])  # 長すぎると困るので頭だけ表示

        # ---- judge ----
        st.subheader("JudgeAI2 の判定結果（llm_meta['judge']）")
        judge = llm_meta.get("judge", {})
        if not judge:
            st.info("judge 情報はまだありません。")
        else:
            st.write(f"- status: `{judge.get('status', 'unknown')}`")
            st.write(f"- chosen_model: `{judge.get('chosen_model', '')}`")

            reason = judge.get("reason")
            if reason:
                with st.expander("選択理由（reason）", expanded=True):
                    st.write(reason)

            chosen_text = (judge.get("chosen_text") or "").strip()
            if chosen_text:
                with st.expander("採用テキスト（chosen_text）", expanded=True):
                    st.text_area(
                        "chosen_text",
                        value=chosen_text,
                        height=260,
                        label_visibility="collapsed",
                    )

            # ▼ 候補モデル＋スコア一覧
            raw_candidates = judge.get("candidates") or []
            with st.expander("候補モデル一覧（candidates / scores）", expanded=False):
                if isinstance(raw_candidates, dict):
                    # もし dict 形式: {model_name: {score, text, ...}}
                    for name, info in raw_candidates.items():
                        score = info.get("score", "-")
                        preview = (info.get("text") or "")[:800]
                        st.markdown(f"### {name}  |  score = `{score}`")
                        st.write(preview)
                        st.markdown("---")
                elif isinstance(raw_candidates, list):
                    # もし list 形式: [{"name":..., "score":..., "text":...}, ...]
                    for i, cand in enumerate(raw_candidates, start=1):
                        name = cand.get("name", f"cand-{i}")
                        score = cand.get("score", "-")
                        length = cand.get("length", 0)
                        preview = (cand.get("text") or "")[:800]
                        st.markdown(
                            f"### 候補 {i}: `{name}`  |  score = `{score}`  |  length = {length}"
                        )
                        st.write(preview)
                        details = cand.get("details") or {}
                        if details:
                            with st.expander("details", expanded=False):
                                st.json(details)
                        st.markdown("---")
                else:
                    st.write("candidates の形式が想定外です:", type(raw_candidates))
    
        # ---- composer ----
        st.subheader("ComposerAI の最終結果（llm_meta['composer']）")
        comp = llm_meta.get("composer", {})
        if not comp:
            st.info("composer 情報はまだありません。")
        else:
            st.write(f"- status: `{comp.get('status', 'unknown')}`")
            st.write(f"- source_model: `{comp.get('source_model', '')}`")
            st.write(f"- mode: `{comp.get('mode', '')}`")

            summary = comp.get("summary")
            if summary:
                with st.expander("サマリ（summary）", expanded=True):
                    st.text_area(
                        "composer_summary",
                        value=str(summary),
                        height=200,
                        label_visibility="collapsed",
                    )

            text = (comp.get("text") or "").strip()
            if text:
                with st.expander("最終返答テキスト（composer.text）", expanded=True):
                    st.text_area(
                        "composer_text",
                        value=text,
                        height=260,
                        label_visibility="collapsed",
                    )

        # ---- MemoryAI ----
        st.subheader("MemoryAI の状態（長期記憶）")
        memory_ctx = llm_meta.get("memory_context") or ""
        mem_update = llm_meta.get("memory_update") or {}

        # AnswerTalker が抱えている MemoryAI インスタンスを取得
        memory_ai = getattr(self.answer_talker, "memory_ai", None)

        if memory_ai is None:
            st.warning("AnswerTalker.memory_ai が初期化されていません。")
        else:
            persona_id = getattr(memory_ai, "persona_id", "default")
            max_records = getattr(memory_ai, "max_store_items", 0)
            storage_file = getattr(memory_ai, "file_path", "(unknown)")
            st.write(f"- persona_id: `{persona_id}`")
            st.write(f"- max_records: `{max_records}`")
            st.write(f"- storage_file: `{storage_file}`")

            # メモリ一覧（MemoryAI.get_all_records）
            try:
                records = memory_ai.get_all_records()
            except Exception as e:
                records = []
                st.warning(f"MemoryRecord の取得に失敗しました: {e}")

            if not records:
                st.info("現在、保存済みの MemoryRecord はありません。")
            else:
                st.markdown("#### 保存済み MemoryRecord 一覧")
                for i, r in enumerate(records, start=1):
                    with st.expander(
                        f"記憶 {i}: [imp={r.importance}] {r.summary[:32]}...",
                        expanded=False,
                    ):
                        st.write(f"- id: `{r.id}`")
                        st.write(f"- round_id: {r.round_id}")
                        st.write(f"- importance: {r.importance}")
                        st.write(f"- created_at: {r.created_at}")
                        st.write(
                            f"- tags: {', '.join(r.tags) if r.tags else '(なし)'}"
                        )
                        st.write("**summary:**")
                        st.write(r.summary)
                        if r.source_user:
                            st.write("\n**source_user:**")
                            st.text(r.source_user)
                        if r.source_assistant:
                            st.write("\n**source_assistant:**")
                            st.text(r.source_assistant)

            # ▼ 実ファイルの診断（JSON が本当に作られているか）
            st.markdown("---")
            st.markdown("### MemoryAI ファイル診断（JSON）")

            if st.button("記憶ファイルを診断する", key="memfile_check_at"):
                path = storage_file
                st.write(f"対象ファイル: `{path}`")

                if not path or path == "(unknown)":
                    st.error("MemoryAI.file_path が正しく設定されていません。")
                elif not os.path.exists(path):
                    st.error(
                        "ファイルが存在しません。まだ一度も記憶が保存されていない可能性があります。"
                    )
                else:
                    st.success("ファイルは存在します。")

                    size = os.path.getsize(path)
                    st.write(f"- ファイルサイズ: `{size}` バイト")

                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    except Exception as e:
                        st.error(f"JSON の読み込みに失敗しました: {e}")
                    else:
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

        st.subheader("llm_meta 内のメモリ関連メタ情報")
        st.write(f"- memory_context:\n\n```text\n{memory_ctx}\n```")
        st.write("- memory_update（直近ターンの記憶更新結果）:")
        st.json(mem_update)


# ===== ここが重要：ModeSwitcher から呼ぶファクトリ =====

def create_answertalker_view() -> AnswerTalkerView:
    """
    ModeSwitcher から呼ぶためのシンプルなファクトリ関数。
    """
    return AnswerTalkerView()
