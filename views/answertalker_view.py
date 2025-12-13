from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional

import os
import json
import streamlit as st

from actors.actor import Actor
from actors.answer_talker import AnswerTalker
from actors.persona.persona_classes.persona_riseria_ja import Persona

# 環境変数でデバッグモードを切り替え
LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"


class AnswerTalkerView:
    """
    AnswerTalker / ModelsAI2 / JudgeAI3 / ComposerAI / MemoryAI の
    デバッグ・閲覧用ビュー。

    重要：
    - このビュー自体は LLM を実行しない。
    - ただし st.session_state["llm_meta"] の中身を見て
      「どこまでパイプラインが進んだか」を診断できるようにする。
    """

    TITLE = "🧩 AnswerTalker（AI統合テスト）"

    def __init__(self) -> None:
        # --- プレイヤー名（UserSettings 由来）を取得 ---
        player_name = st.session_state.get("player_name", "アツシ")

        # Persona には player_name を渡しておく
        persona = Persona(player_name=player_name)

        # Actor（このビューでは speak() は呼ばないが、AnswerTalkerの初期化整合のため保持）
        self.actor = Actor("floria", persona)

        # ★ Streamlit の state を AnswerTalker に明示的に渡す（デバッグ時）
        #   これにより、AnswerTalker 内部とビューの両方で llm_meta を確実に共有できる。
        state: Optional[MutableMapping[str, Any]] = st.session_state if LYRA_DEBUG else None

        self.answer_talker = AnswerTalker(
            persona,
            state=state,
        )

    # =========================================================
    # Helpers
    # =========================================================
    @staticmethod
    def _safe_json(obj: Any) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            return str(obj)

    @staticmethod
    def _pick_error_keys(llm_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        llm_meta のうち error 系（*_error / *Error / errors）だけ抽出。
        """
        out: Dict[str, Any] = {}
        for k, v in (llm_meta or {}).items():
            lk = str(k).lower()
            if lk.endswith("_error") or lk.endswith("error") or lk == "errors":
                if v:
                    out[k] = v
        return out

    # =========================================================
    # Render
    # =========================================================
    def render(self) -> None:
        st.header(self.TITLE)

        # 現在の軽い表示
        player_name = st.session_state.get("player_name", "アツシ")
        reply_length_mode = st.session_state.get("reply_length_mode", "auto")
        st.caption(
            f"現在のプレイヤー名: **{player_name}**  /  "
            f"発話長さモード: **{reply_length_mode}**"
        )

        st.info(
            "この画面では、Actor に紐づく AnswerTalker が保持している llm_meta の内容 "
            "（system_prompt / emotion_override / models / judge / composer / emotion / memory）を参照できます。\n\n"
            "※ この画面から LLM を実行することはありません。"
        )

        llm_meta: Dict[str, Any] = st.session_state.get("llm_meta", {}) or {}

        # -----------------------------------------------------
        # 0) パイプライン診断（Errors / 状態）
        # -----------------------------------------------------
        st.subheader("🧯 パイプライン診断（Errors / 状態）")

        err_pack = self._pick_error_keys(llm_meta)
        if not err_pack:
            st.success("現在、明示的な error キーは検出されていません。")
        else:
            st.error("error キーが検出されています。下を確認してください。")
            st.json(err_pack)

        # keys 一覧（上位40）
        keys = list(llm_meta.keys())
        with st.expander("llm_meta keys（上位40）", expanded=True):
            st.json(keys[:40])

        # ModelsAI2 が「呼ばれたのに空」を見抜くヒント表示
        emo_override = llm_meta.get("emotion_override") or {}
        models = llm_meta.get("models", {})

        if emo_override and not models:
            st.warning(
                "⚠️ emotion_override は入っていますが、llm_meta['models'] が空です。\n"
                "つまり『MixerAI までは動いたが、ModelsAI2.collect の結果が空（または書き戻しが失敗）』の可能性が高いです。"
            )

        # LLMManager 診断（AnswerTalker 側の llm_manager が取れれば表示）
        with st.expander("LLMManager 診断（モデル設定の有無）", expanded=False):
            lm = getattr(self.answer_talker, "llm_manager", None)
            if lm is None:
                st.warning("AnswerTalker.llm_manager が見つかりません。")
            else:
                try:
                    props = lm.get_model_props()
                except Exception as e:
                    st.error(f"get_model_props() で例外: {e}")
                    props = {}
                st.write(f"- model_props keys: {list(props.keys())[:20]}")
                st.json(props)

        st.markdown("---")

        # -----------------------------------------------------
        # 1) system_prompt_used
        # -----------------------------------------------------
        st.subheader("今回使用された system_prompt（emotion 反映後）")
        sys_used = llm_meta.get("system_prompt_used") or ""
        if not sys_used:
            st.info("system_prompt_used はまだありません。")
        else:
            st.text_area(
                "system_prompt_used",
                value=sys_used,
                height=240,
                label_visibility="collapsed",
            )

        st.markdown("---")

        # -----------------------------------------------------
        # 2) emotion_override
        # -----------------------------------------------------
        st.subheader("emotion_override（MixerAI → ModelsAI2）")
        if not emo_override:
            st.info("emotion_override はまだありません。")
        else:
            st.json(emo_override)

        st.markdown("---")

        # -----------------------------------------------------
        # 3) models（ModelsAI2 output）
        # -----------------------------------------------------
        st.subheader("llm_meta['models']（ModelsAI2 の出力）")

        if not models:
            st.warning("models は空です（collect は呼ばれたが結果なし / 書き込みなしの可能性）")
        else:
            for name, info in models.items():
                with st.expander(f"モデル: {name}", expanded=True):
                    status = (info or {}).get("status", "unknown")
                    text = (info or {}).get("text", "") or ""
                    usage = (info or {}).get("usage")
                    error = (info or {}).get("error")

                    st.write("- status:", status)
                    st.write("- len(text):", len(text))
                    if usage is not None:
                        st.write("- usage:", usage)
                    if error:
                        st.error(f"error: {error}")
                    if text:
                        st.markdown("**preview:**")
                        st.code(text[:1200])

        st.markdown("---")

        # -----------------------------------------------------
        # 4) judge
        # -----------------------------------------------------
        st.subheader("JudgeAI3（llm_meta['judge']）")
        judge = llm_meta.get("judge", {}) or {}
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
                    for n, inf in raw_candidates.items():
                        score = inf.get("score", "-")
                        preview = (inf.get("text") or "")[:800]
                        st.markdown(f"### {n}  |  score = `{score}`")
                        st.write(preview)
                        st.markdown("---")
                elif isinstance(raw_candidates, list):
                    for i, cand in enumerate(raw_candidates, start=1):
                        n = cand.get("name", f"cand-{i}")
                        score = cand.get("score", "-")
                        length = cand.get("length", 0)
                        preview = (cand.get("text") or "")[:800]
                        st.markdown(
                            f"### 候補 {i}: `{n}`  |  score = `{score}`  |  length = {length}"
                        )
                        details = cand.get("details") or {}
                        if details:
                            with st.expander("details", expanded=False):
                                st.json(details)
                        st.markdown("---")
                else:
                    st.write("candidates の形式が想定外です:", type(raw_candidates))

        st.markdown("---")

        # -----------------------------------------------------
        # 5) composer
        # -----------------------------------------------------
        st.subheader("ComposerAI（llm_meta['composer']）")
        comp = llm_meta.get("composer", {}) or {}
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

        st.markdown("---")

        # -----------------------------------------------------
        # 6) judge mode 状態
        # -----------------------------------------------------
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

        st.markdown("---")

        # -----------------------------------------------------
        # 7) EmotionAI
        # -----------------------------------------------------
        st.subheader("EmotionAI（llm_meta['emotion']）")
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

        st.markdown("---")

        # -----------------------------------------------------
        # 8) MemoryAI
        # -----------------------------------------------------
        st.subheader("MemoryAI（長期記憶）")

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
                        st.write(f"- tags: {', '.join(r.tags) if r.tags else '(なし)'}")
                        st.write("**summary:**")
                        st.write(r.summary)
                        if r.source_user:
                            st.write("\n**source_user:**")
                            st.text(r.source_user)
                        if r.source_assistant:
                            st.write("\n**source_assistant:**")
                            st.text(r.source_assistant)

        st.subheader("llm_meta 内のメモリ関連メタ情報")
        st.write(f"- memory_context:\n\n```text\n{memory_ctx}\n```")
        st.write("- memory_update（直近ターンの記憶更新結果）:")
        st.json(mem_update)


def create_answertalker_view() -> AnswerTalkerView:
    """
    ModeSwitcher から呼ぶためのシンプルなファクトリ関数。
    """
    return AnswerTalkerView()
