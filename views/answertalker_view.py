# views/answertalker_view.py
from __future__ import annotations

from typing import Any, Dict, MutableMapping, List
import os
import json
import streamlit as st

from auth.roles import Role  # いまは未使用だが将来の拡張用に残しておく
from actors.actor import Actor
from actors.answer_talker import AnswerTalker
from actors.persona.persona_classes.persona_riseria_ja import Persona

LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"


class AnswerTalkerView:
    """
    AnswerTalker / ModelsAI / JudgeAI3 / ComposerAI / MemoryAI のデバッグ・閲覧用ビュー（閲覧専用）
    """

    TITLE = "🧩 AnswerTalker（AI統合テスト）"

    @staticmethod
    def _render_any_as_textarea(label: str, value: Any, height: int = 220) -> None:
        if isinstance(value, str):
            st.text_area(label, value=value, height=height, label_visibility="collapsed")
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

    # =========================================================
    # World Change（importance=5）表示用ヘルパ
    # =========================================================
    @staticmethod
    def _label_reason_unavailable(code: Any) -> str:
        s = "" if code is None else str(code)
        if s == "interpersonal_complexity":
            return "🤝 対人関係（複合）"
        if s == "external_event":
            return "🌪 外的要因（天変地異/不可抗力）"
        if not s:
            return "(なし)"
        return f"(unknown: {s})"

    @staticmethod
    def _container_border():
        # streamlit のバージョン差で container(border=...) が無い環境でも落とさない
        try:
            return st.container(border=True)
        except Exception:
            return st.container()

    @staticmethod
    def _render_world_change_records(records: List[Any]) -> None:
        wc: List[Any] = []
        for r in records:
            try:
                if int(getattr(r, "importance", 0) or 0) >= 5:
                    wc.append(r)
            except Exception:
                continue

        try:
            wc.sort(key=lambda x: getattr(x, "created_at", "") or "", reverse=True)
        except Exception:
            pass

        st.markdown("### 🌍 世界変化記憶（importance=5）")
        if not wc:
            st.info("世界変化記憶はまだありません。")
            return

        for i, r in enumerate(wc, start=1):
            summary = getattr(r, "summary", "") or ""
            created_at = getattr(r, "created_at", "") or ""
            rid = getattr(r, "round_id", None)

            with AnswerTalkerView._container_border():
                st.markdown(f"**{i}. {summary}**")
                st.caption(f"created_at: {created_at} / round_id: {rid}")

                tags = getattr(r, "tags", []) or []
                if tags:
                    st.write("Tags:", ", ".join([str(x) for x in tags]))

                reasons = getattr(r, "world_change_reasons", None)
                if isinstance(reasons, list) and reasons:
                    st.write("**Triggered by:**")
                    for t in reasons[:5]:
                        st.markdown(f"- {t}")
                else:
                    rnu = getattr(r, "reason_unavailable", None)
                    st.write("**Reason unavailable:**", AnswerTalkerView._label_reason_unavailable(rnu))

                with st.expander("Source (raw)", expanded=False):
                    su = getattr(r, "source_user", "") or ""
                    sa = getattr(r, "source_assistant", "") or ""
                    if su:
                        st.markdown("**source_user:**")
                        st.text(su)
                    if sa:
                        st.markdown("**source_assistant:**")
                        st.text(sa)

    def __init__(self) -> None:
        player_name = st.session_state.get("player_name", "アツシ")

        persona = Persona(player_name=player_name)
        self.actor = Actor("floria", persona)

        # ★閲覧専用：session_state を AnswerTalker に渡さない
        local_state: MutableMapping[str, Any] = {}

        self.answer_talker = AnswerTalker(
            persona,
            state=local_state,
        )

    def render(self) -> None:
        st.header(self.TITLE)

        player_name = st.session_state.get("player_name", "アツシ")
        reply_length_mode = st.session_state.get("reply_length_mode", "auto")
        st.caption(f"現在のプレイヤー名: **{player_name}**  /  発話長さモード: **{reply_length_mode}**")

        st.info(
            "この画面では llm_meta の内容（system_prompt / emotion_override / models / judge / composer / emotion / memory）を参照できます。\n\n"
            "※ この画面から speak() や MemoryAI.update_from_turn() は実行しません。"
        )

        llm_meta: Dict[str, Any] = st.session_state.get("llm_meta", {}) or {}

        st.subheader("今回使用された system_prompt（affection / ドキドキ💓反映後）")
        if "system_prompt_used" not in llm_meta:
            st.info("system_prompt_used はまだありません。（キー未作成）")
        else:
            sys_used = llm_meta.get("system_prompt_used")
            st.caption(
                f"system_prompt_used type={type(sys_used).__name__} / "
                f"len={len(sys_used) if isinstance(sys_used, str) else '(n/a)'}"
            )
            self._render_any_as_textarea("system_prompt_used", sys_used, height=220)

        st.subheader("emotion_override（MixerAI → ModelsAI に渡した感情オーバーライド）")
        emo_override = llm_meta.get("emotion_override") or {}
        if not emo_override:
            st.info("emotion_override はまだありません。")
        else:
            st.json(emo_override)

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
                    st.text_area("chosen_text", value=chosen_text, height=260, label_visibility="collapsed")

        st.subheader("ComposerAI の最終結果（llm_meta['composer']）")
        comp = llm_meta.get("composer", {})
        if not comp:
            st.info("composer 情報はまだありません。")
        else:
            st.write(f"- status: `{comp.get('status', 'unknown')}`")
            st.write(f"- source_model: `{comp.get('source_model', '')}`")
            st.write(f"- mode: `{comp.get('mode', '')}`")
            st.write(f"- is_modified: `{comp.get('is_modified', False)}`")

            base_text = (comp.get("base_text") or "").strip()
            final_text = (comp.get("text") or "").strip()

            if base_text:
                with st.expander("Refiner 前のテキスト（base_text）", expanded=False):
                    st.text_area("composer_base_text", value=base_text, height=260, label_visibility="collapsed")
            if final_text:
                with st.expander("最終返答テキスト（composer.text）", expanded=True):
                    st.text_area("composer_text", value=final_text, height=260, label_visibility="collapsed")

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

        st.subheader("MemoryAI の状態（長期記憶）")
        memory_ai = getattr(self.answer_talker, "memory_ai", None)

        if memory_ai is None:
            st.warning("AnswerTalker.memory_ai が初期化されていません。")
            return

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
            self._render_world_change_records(records)
            st.markdown("---")

            st.markdown("#### 保存済み MemoryRecord 一覧（全件）")
            for i, r in enumerate(records, start=1):
                imp = getattr(r, "importance", 0)
                summ = getattr(r, "summary", "") or ""
                summ_head = (summ[:32] + "...") if len(summ) > 32 else summ

                with st.expander(f"記憶 {i}: [imp={imp}] {summ_head}", expanded=False):
                    st.write(f"- id: `{getattr(r, 'id', '')}`")
                    st.write(f"- round_id: {getattr(r, 'round_id', 0)}")
                    st.write(f"- importance: {imp}")
                    st.write(f"- created_at: {getattr(r, 'created_at', '')}")
                    tags = getattr(r, "tags", None) or []
                    st.write(f"- tags: {', '.join(tags) if tags else '(なし)'}")

                    if int(imp or 0) >= 5:
                        wcr = getattr(r, "world_change_reasons", None)
                        rnu = getattr(r, "reason_unavailable", None)
                        st.markdown("**world_change:**")
                        if isinstance(wcr, list) and wcr:
                            st.write("- world_change_reasons:")
                            st.json(wcr)
                        else:
                            st.write("- reason_unavailable:", self._label_reason_unavailable(rnu))

                    st.write("**summary:**")
                    st.write(summ)

                    su = getattr(r, "source_user", "") or ""
                    sa = getattr(r, "source_assistant", "") or ""
                    if su:
                        st.write("\n**source_user:**")
                        st.text(su)
                    if sa:
                        st.write("\n**source_assistant:**")
                        st.text(sa)


def create_answertalker_view() -> AnswerTalkerView:
    return AnswerTalkerView()
