from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional

import os
import json
import streamlit as st

from actors.actor import Actor
from actors.answer_talker import AnswerTalker
from actors.persona.persona_classes.persona_riseria_ja import Persona


# ==========================================================
# Debug flag
# ==========================================================
LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"


class AnswerTalkerView:
    """
    AnswerTalker / ModelsAI / JudgeAI3 / ComposerAI / MemoryAI の
    デバッグ・閲覧用ビュー。

    ★ 重要:
      - AnswerTalker が session_state["llm_meta"] に積んだ情報を
        そのまま可視化する
      - ModelsAI2 が「呼ばれていない」「例外で落ちている」
        「空 dict を返している」をすべて判別できるようにする
    """

    TITLE = "🧩 AnswerTalker（AI統合テスト）"

    def __init__(self) -> None:
        # ---- プレイヤー名 ----
        player_name = st.session_state.get("player_name", "アツシ")

        # ---- Persona / Actor ----
        persona = Persona(player_name=player_name)
        self.actor = Actor("riseria", persona)

        # ---- state を共有（重要）----
        state: Optional[MutableMapping[str, Any]]
        state = st.session_state

        # ---- AnswerTalker ----
        self.answer_talker = AnswerTalker(
            persona=persona,
            state=state,
        )

    # ======================================================
    # Render
    # ======================================================
    def render(self) -> None:
        st.header(self.TITLE)

        # ---- 基本情報 ----
        player_name = st.session_state.get("player_name", "アツシ")
        reply_length_mode = st.session_state.get("reply_length_mode", "auto")

        st.caption(
            f"現在のプレイヤー名: **{player_name}** / "
            f"発話長さモード: **{reply_length_mode}**"
        )

        st.info(
            "この画面では、Actor に紐づく AnswerTalker が保持している llm_meta の内容 "
            "（system_prompt / emotion_override / models / judge / composer / emotion / memory）を参照できます。\n\n"
            "※ この画面から LLM を実行することはありません。"
        )

        llm_meta: Dict[str, Any] = st.session_state.get("llm_meta", {}) or {}

        # ==================================================
        # ★ 最重要：パイプライン診断
        # ==================================================
        st.subheader("🚨 パイプライン診断（Errors / 状態）")

        err_keys = [
            "models_error",
            "models_trace",
            "judge_error",
            "composer_error",
            "emotion_override_error",
            "world_error",
            "emotion_error",
        ]

        found_error = False
        for k in err_keys:
            v = llm_meta.get(k)
            if v:
                found_error = True
                if "trace" in k:
                    with st.expander(f"{k}（trace）", expanded=False):
                        st.code(str(v))
                else:
                    st.error(f"{k}: {v}")

        if not found_error:
            st.success("現在、明示的な error キーは検出されていません。")

        st.caption("llm_meta keys（上位40）")
        st.write(sorted(list(llm_meta.keys()))[:40])

        st.markdown("---")

        # ==================================================
        # system_prompt_used
        # ==================================================
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

        # ==================================================
        # emotion_override
        # ==================================================
        st.subheader("emotion_override（MixerAI → ModelsAI2）")
        emo_override = llm_meta.get("emotion_override") or {}
        if not emo_override:
            st.info("emotion_override はまだありません。")
        else:
            st.json(emo_override)

        # ==================================================
        # models
        # ==================================================
        st.subheader("llm_meta['models']（ModelsAI2 の生出力）")
        models = llm_meta.get("models")

        if models is None:
            st.warning("models キー自体が存在しません（ModelsAI2.collect 未実行）")
        elif not models:
            st.warning("models は空です（collect は呼ばれたが結果なし）")
        else:
            for name, info in models.items():
                with st.expander(f"モデル: {name}", expanded=True):
                    st.write("- status:", info.get("status"))
                    st.write("- len(text):", len(info.get("text", "") or ""))
                    if info.get("error"):
                        st.error(info["error"])
                    if info.get("text"):
                        st.code(info["text"][:1500])

        # ==================================================
        # judge
        # ==================================================
        st.subheader("JudgeAI3（llm_meta['judge']）")
        judge = llm_meta.get("judge")

        if not judge:
            st.info("judge 情報はまだありません。")
        else:
            st.json(judge)

        # ==================================================
        # composer
        # ==================================================
        st.subheader("ComposerAI（llm_meta['composer']）")
        composer = llm_meta.get("composer")

        if not composer:
            st.info("composer 情報はまだありません。")
        else:
            st.json(composer)

        # ==================================================
        # emotion
        # ==================================================
        st.subheader("EmotionAI（llm_meta['emotion']）")
        emotion = llm_meta.get("emotion")

        if not emotion:
            st.info("emotion 情報はまだありません。")
        else:
            st.json(emotion)

        # ==================================================
        # memory
        # ==================================================
        st.subheader("MemoryAI")
        mem_ctx = llm_meta.get("memory_context") or ""
        mem_update = llm_meta.get("memory_update") or {}

        if mem_ctx:
            with st.expander("memory_context", expanded=False):
                st.text(mem_ctx)

        if mem_update:
            with st.expander("memory_update", expanded=False):
                st.json(mem_update)


# ==========================================================
# Factory
# ==========================================================
def create_answertalker_view() -> AnswerTalkerView:
    return AnswerTalkerView()
