# views/answertalker_view.py
from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional

import os
import json
import streamlit as st

from auth.roles import Role  # いまは未使用だが将来の拡張用に残しておく
from actors.actor import Actor
from actors.answer_talker import AnswerTalker
from actors.persona.persona_classes.persona_riseria_ja import Persona


# 環境変数でデバッグモードを切り替え
LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"


class AnswerTalkerView:
    """
    AnswerTalker / ModelsAI / JudgeAI3 / ComposerAI / MemoryAI の
    デバッグ・閲覧用ビュー。

    注意：
    - このビューは「閲覧専用」。
    - AnswerTalker は InitAI 経由で session_state を補修/初期化しうるため、
      ここから st.session_state を AnswerTalker に渡すと
      “見るだけのつもりが状態を書き換える” 副作用が起きる。
    - したがって AnswerTalker にはローカル state を渡し、
      st.session_state の llm_meta を安全に閲覧する。
    """

    TITLE = "🧩 AnswerTalker（AI統合テスト）"

    @staticmethod
    def _render_any_as_textarea(label: str, value: Any, height: int = 220) -> None:
        """
        llm_meta は過渡的に型が変わることがある。
        - str: text_area
        - dict/list: pretty json を text_area
        - その他: str() を text_area
        """
        if isinstance(value, str):
            st.text_area(
                label,
                value=value,
                height=height,
                label_visibility="collapsed",
            )
            return

        if isinstance(value, (dict, list)):
            st.text_area(
                label,
                value=json.dumps(value, ensure_ascii=False, indent=2),
                height=height,
                label_visibility="collapsed",
            )
            return

        st.text_area(
            label,
            value="" if value is None else str(value),
            height=height,
            label_visibility="collapsed",
        )

    def __init__(self) -> None:
        # --- プレイヤー名（UserSettings 由来）を取得 ---
        player_name = st.session_state.get("player_name", "アツシ")

        # Actor と AnswerTalker を初期化
        # Persona には player_name を渡しておく
        persona = Persona(player_name=player_name)
        self.actor = Actor("floria", persona)

        # ★重要：閲覧専用ビューなので session_state を AnswerTalker に渡さない
        #   → AnswerTalker 生成による session_state への副作用を遮断する
        local_state: MutableMapping[str, Any] = {}

        self.answer_talker = AnswerTalker(
            persona,
            state=local_state,
        )

    def render(self) -> None:
        st.header(self.TITLE)

        # 現在のユーザー設定の軽い表示（任意・デバッグ補助）
        player_name = st.session_state.get("player_name", "アツシ")
        reply_length_mode = st.session_state.get("reply_length_mode", "auto")
        st.caption(
            f"現在のプレイヤー名: **{player_name}**  /  "
            f"発話長さモード: **{reply_length_mode}**"
        )

        st.info(
            "この画面では、Actor に紐づく AnswerTalker が保持している llm_meta の内容 "
            "（system_prompt / emotion_override / models / judge / composer / emotion / memory）を参照できます。\n\n"
            "※ この画面からは AnswerTalker.run_models() や MemoryAI.update_from_turn() などは実行しません。"
        )

        llm_meta: Dict[str, Any] = st.session_state.get("llm_meta", {}) or {}

        # ---- 今回使用された system_prompt ----
        st.subheader("今回使用された system_prompt（affection / ドキドキ💓反映後）")

        if "system_prompt_used" not in llm_meta:
            st.info("system_prompt_used はまだありません。（キー未作成）")
        else:
            sys_used = llm_meta.get("system_prompt_used")
            st.caption(
                f"system_prompt_used type={type(sys_used).__name__} / "
                f"len={len(sys_used) if isinstance(sys_used, str) else '(n/a)'}"
            )
            # 空文字でも「空が入っている」こと自体が重要なので必ず出す
            self._render_any_as_textarea("system_prompt_used", sys_used, height=220)

        # ---- emotion_override ----
        st.subheader("emotion_override（MixerAI → ModelsAI に渡した感情オーバーライド）")
        emo_override = llm_meta.get("emotion_override") or {}
        if not emo_override:
            st.info("emotion_override はまだありません。")
        else:
            st.json(emo_override)

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

                    if text:
                        st.markdown("**preview:**")
                        st.code(text[:1000])

        # ---- judge ----
        st.subheader("JudgeAI3 の判定結果（llm_meta['judge']）")
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

            raw_candidates = judge.get("candidates") or []
            with st.expander("候補モデル一覧（candidates / scores）", expanded=False):
                if isinstance(raw_candidates, dict):
                    for cand_name, cand_info in raw_candidates.items():
                        score = cand_info.get("score", "-")
                        preview = (cand_info.get("text") or "")[:800]
                        st.markdown(f"### {cand_name}  |  score = `{score}`")
                        st.write(preview)
                        st.markdown("---")
                elif isinstance(raw_candidates, list):
                    for i, cand in enumerate(raw_candidates, start=1):
                        cand_name = cand.get("name", f"cand-{i}")
                        score = cand.get("score", "-")
                        length = cand.get("length", 0)
                        preview = (cand.get("text") or "")[:800]
                        st.markdown(
                            f"### 候補 {i}: `{cand_name}`  |  score = `{score}`  |  length = {length}"
                        )
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

            base_src = comp.get("base_source_model")
            if base_src:
                st.write(f"- base_source_model: `{base_src}`")

            dev_force = comp.get("dev_force_model")
            if dev_force:
                st.write(f"- dev_force_model: `{dev_force}`")

            st.write(f"- is_modified: `{comp.get('is_modified', False)}`")

            summary = comp.get("summary")
            if summary:
                with st.expander("サマリ（summary）", expanded=True):
                    st.text_area(
                        "composer_summary",
                        value=str(summary),
                        height=200,
                        label_visibility="collapsed",
                    )

            # Refiner 情報
            with st.expander("Refiner 情報", expanded=False):
                st.write(f"- refiner_model: `{comp.get('refiner_model', None)}`")
                st.write(f"- refiner_used: `{comp.get('refiner_used', False)}`")
                st.write(f"- refiner_status: `{comp.get('refiner_status', '')}`")
                ref_err = comp.get("refiner_error")
                if ref_err:
                    st.error(f"refiner_error: {ref_err}")

            # base_text / 最終テキスト比較
            base_text = (comp.get("base_text") or "").strip()
            final_text = (comp.get("text") or "").strip()

            if base_text:
                with st.expander("Refiner 前のテキスト（base_text）", expanded=False):
                    st.text_area(
                        "composer_base_text",
                        value=base_text,
                        height=260,
                        label_visibility="collapsed",
                    )

            if final_text:
                with st.expander("最終返答テキスト（composer.text）", expanded=True):
                    st.text_area(
                        "composer_text",
                        value=final_text,
                        height=260,
                        label_visibility="collapsed",
                    )

        # ---- Composer 用スタイルヒント（persona 由来） ----
        style_hint = llm_meta.get("composer_style_hint") or ""
        if style_hint:
            st.subheader("Composer 用スタイルヒント（persona 由来）")
            with st.expander("composer_style_hint", expanded=False):
                st.text_area(
                    "composer_style_hint",
                    value=style_hint,
                    height=260,
                    label_visibility="collapsed",
                )

        # ---- Judge モード状態 ----
        st.subheader("Judge モード状態")
        current_mode_meta = llm_meta.get("judge_mode", None)
        next_mode_meta = llm_meta.get("judge_mode_next", None)
        session_mode = st.session_state.get("judge_mode", None)

        cols_mode = st.columns(3)
        with cols_mode[0]:
            st.write(f"llm_meta['judge_mode']: `{current_mode_meta}`")
        with cols_mode[1]:
            st.write(f"llm_meta['judge_mode_next']: `{next_mode_meta}`")
        with cols_mode[2]:
            st.write(f"session_state['judge_mode']: `{session_mode}`")

        # ---- EmotionAI ----
        st.subheader("EmotionAI の解析結果（llm_meta['emotion']）")

        emo = llm_meta.get("emotion") or {}
        emo_err = llm_meta.get("emotion_error")

        if emo_err:
            st.error(f"EmotionAI error: {emo_err}")

        if not emo:
            st.info("Emotion 情報はまだありません。")
        else:
            st.markdown(f"- 推定 judge_mode: `{emo.get('mode', 'normal')}`")

            cols = st.columns(3)
            with cols[0]:
                st.write(f"affection: {emo.get('affection', 0.0):.2f}")
                st.write(f"arousal:   {emo.get('arousal', 0.0):.2f}")
            with cols[1]:
                st.write(f"tension:   {emo.get('tension', 0.0):.2f}")
                st.write(f"anger:     {emo.get('anger', 0.0):.2f}")
            with cols[2]:
                st.write(f"sadness:   {emo.get('sadness', 0.0):.2f}")
                st.write(f"excitement:{emo.get('excitement', 0.0):.2f}")

            with st.expander("raw_text（EmotionAI の LLM 出力）", expanded=False):
                st.code(emo.get("raw_text", ""), language="json")

        # ---- MemoryAI ----
        st.subheader("MemoryAI の状態（長期記憶）")
        memory_ctx = llm_meta.get("memory_context") or ""
        mem_update = llm_meta.get("memory_update") or {}

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


def create_answertalker_view() -> AnswerTalkerView:
    """
    ModeSwitcher から呼ぶためのシンプルなファクトリ関数。
    """
    return AnswerTalkerView()
