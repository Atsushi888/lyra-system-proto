# components/emotion_control.py
from __future__ import annotations

from typing import Any, Dict

import streamlit as st


class EmotionControl:
    """
    感情オーバーライド用のコントロールパネル。

    - emotion_override_mode:
        "auto"        → EmotionAI の結果をそのまま利用
        "manual_full" → このパネルの値で完全上書き（EmotionAI 無視）

    - emotion_override_manual:
        {
          "mode": "normal" | "erotic" | "debate",
          "affection": float,
          "arousal": float,
          "tension": float,
          "anger": float,
          "sadness": float,
          "excitement": float,
        }
      を st.session_state に保存する。
    """

    def __init__(self, session_prefix: str = "emotion_override") -> None:
        self.session_prefix = session_prefix

    # -----------------------------
    def _get_manual_defaults(self) -> Dict[str, Any]:
        saved = st.session_state.get("emotion_override_manual")
        if isinstance(saved, dict):
            return saved

        # 初期値（ちょい好意・ちょいワクワク）
        return {
            "mode": "normal",
            "affection": 0.6,
            "arousal": 0.2,
            "tension": 0.2,
            "anger": 0.0,
            "sadness": 0.1,
            "excitement": 0.5,
        }

    # -----------------------------
    def render(self) -> None:
        st.markdown("## 🌸 感情オーバーライド設定")

        # -------------------------
        # モード切替
        # -------------------------
        mode = st.radio(
            "オーバーライドモード",
            options=["auto", "manual_full"],
            index=0 if st.session_state.get("emotion_override_mode", "auto") == "auto" else 1,
            format_func=lambda x: "自動（EmotionAI の判定に任せる）"
            if x == "auto"
            else "手動で完全上書きする",
            key="emotion_override_mode",
        )

        st.caption(
            "- **自動**: EmotionAI が推定した感情状態を LLM 側に渡します。\n"
            "- **手動で完全上書き**: 下のスライダー値だけを LLM に渡し、EmotionAI の数値は無視します。"
        )

        # -------------------------
        # 手動値スライダー
        # -------------------------
        defaults = self._get_manual_defaults()

        st.markdown("### 手動感情パラメータ")
        col_mode, col_dummy = st.columns([1, 2])
        with col_mode:
            manual_mode = st.selectbox(
                "会話モード",
                options=["normal", "erotic", "debate"],
                index=["normal", "erotic", "debate"].index(defaults.get("mode", "normal")),
            )

        col1, col2 = st.columns(2)
        with col1:
            affection = st.slider(
                "好意 / Affection",
                0.0,
                1.0,
                float(defaults.get("affection", 0.6)),
                0.05,
            )
            arousal = st.slider(
                "性的な高ぶり / Arousal",
                0.0,
                1.0,
                float(defaults.get("arousal", 0.2)),
                0.05,
            )
            excitement = st.slider(
                "ワクワク / Excitement",
                0.0,
                1.0,
                float(defaults.get("excitement", 0.5)),
                0.05,
            )

        with col2:
            tension = st.slider(
                "緊張 / Tension",
                0.0,
                1.0,
                float(defaults.get("tension", 0.2)),
                0.05,
            )
            anger = st.slider(
                "怒り / Anger",
                0.0,
                1.0,
                float(defaults.get("anger", 0.0)),
                0.05,
            )
            sadness = st.slider(
                "悲しみ / Sadness",
                0.0,
                1.0,
                float(defaults.get("sadness", 0.1)),
                0.05,
            )

        manual_dict: Dict[str, Any] = {
            "mode": manual_mode,
            "affection": float(affection),
            "arousal": float(arousal),
            "tension": float(tension),
            "anger": float(anger),
            "sadness": float(sadness),
            "excitement": float(excitement),
        }

        # session_state に保存（AnswerTalker から参照）
        st.session_state["emotion_override_manual"] = manual_dict

        st.markdown("#### 現在の手動設定")
        st.json(manual_dict, expanded=False)

        # -------------------------
        # EmotionAI 側の現在値（参考表示）
        # -------------------------
        llm_meta = st.session_state.get("llm_meta", {})
        emo_current = llm_meta.get("emotion") or {}

        with st.expander("EmotionAI が推定した最新の感情状態（参考）", expanded=False):
            if emo_current:
                st.json(emo_current)
            else:
                st.caption("まだ EmotionAI の解析結果がありません。")
