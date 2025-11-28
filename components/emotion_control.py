# components/emotion_control.py
from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from actors.emotion_ai import EmotionResult

# session_state のキーを定数として定義
EMOTION_OVERRIDE_MODE_KEY = "emotion_override_mode"      # "auto" / "manual_full"
EMOTION_OVERRIDE_MANUAL_KEY = "emotion_override_manual"  # Dict[str, Any] を想定


class EmotionControl:
    """
    感情オーバーライド用のコントロールパネル。

    - override モードの選択（auto / manual_full）
    - manual_full のときに各感情値を入力して session_state に保存

    既存ビュー側からは EmotionControl().render() を呼ぶだけでよい想定。
    互換性確保のために __init__ は任意の引数を受け付ける。
    """

    def __init__(self, *_, **__) -> None:
        # 互換性用。特別な初期化は不要。
        pass

    def render(self) -> None:
        st.markdown("### 感情オーバーライド設定")

        # ----- モード選択 -----
        current_mode = st.session_state.get(EMOTION_OVERRIDE_MODE_KEY, "auto")
        mode = st.radio(
            "モード",
            options=["auto", "manual_full"],
            format_func=lambda m: "自動（EmotionAI の結果を使用）"
            if m == "auto"
            else "手動（完全上書き）",
            index=0 if current_mode == "auto" else 1,
            horizontal=False,
            key="emotion_override_mode_radio",
        )
        st.session_state[EMOTION_OVERRIDE_MODE_KEY] = mode

        # ----- manual_full のときだけ詳細パネル -----
        if mode == "manual_full":
            st.caption("手動入力した感情値で、EmotionAI の結果を完全に上書きします。")

            manual: Dict[str, Any] = st.session_state.get(
                EMOTION_OVERRIDE_MANUAL_KEY, {}
            )
            mode_str = manual.get("mode", "normal")

            col1, col2 = st.columns(2)
            with col1:
                mode_str = st.text_input("キャラモード", value=mode_str)

            # スライダー群
            def slider_value(key: str, label: str, default: float = 0.0) -> float:
                cur = float(manual.get(key, default))
                return st.slider(
                    label,
                    min_value=-1.0,
                    max_value=1.0,
                    value=float(cur),
                    step=0.05,
                    key=f"emotion_manual_{key}",
                )

            affection = slider_value("affection", "好意 / 親しみ", 0.0)
            arousal = slider_value("arousal", "興奮度（性的／情動）", 0.0)
            tension = slider_value("tension", "緊張・不安", 0.0)
            anger = slider_value("anger", "怒り", 0.0)
            sadness = slider_value("sadness", "悲しみ", 0.0)
            excitement = slider_value("excitement", "期待・ワクワク", 0.0)

            manual_updated: Dict[str, Any] = {
                "mode": mode_str or "normal",
                "affection": affection,
                "arousal": arousal,
                "tension": tension,
                "anger": anger,
                "sadness": sadness,
                "excitement": excitement,
            }
            st.session_state[EMOTION_OVERRIDE_MANUAL_KEY] = manual_updated
        else:
            # auto のときは manual 設定は残しておくが、特に UI は出さない
            st.caption("EmotionAI が推定した短期感情をそのまま使用します。")


def build_emotion_override_for_models(
    emotion_ai: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """
    ModelsAI2.collect() に渡す emotion_override を構築する共通ヘルパ。

    emotion_override_mode:
      - "auto"        : EmotionAI の短期感情をそのまま渡す
      - "manual_full" : 手動パネルの値で完全上書き（EmotionAI は無視）

    戻り値:
      - None      : override なし
      - Dict[str, Any]:
          {
            "mode": "normal" / "erotic" / ...,
            "affection": float,
            "arousal": float,
            "tension": float,
            "anger": float,
            "sadness": float,
            "excitement": float,
          }
    """

    # モードと手動パラメータを session_state から取得
    mode = st.session_state.get(EMOTION_OVERRIDE_MODE_KEY, "auto")
    manual = st.session_state.get(EMOTION_OVERRIDE_MANUAL_KEY)

    # ---------------------------
    # 1) 手動完全上書きモード
    # ---------------------------
    if mode == "manual_full":
        if isinstance(manual, dict):
            return {
                "mode": manual.get("mode", "normal"),
                "affection": float(manual.get("affection", 0.0)),
                "arousal": float(manual.get("arousal", 0.0)),
                "tension": float(manual.get("tension", 0.0)),
                "anger": float(manual.get("anger", 0.0)),
                "sadness": float(manual.get("sadness", 0.0)),
                "excitement": float(manual.get("excitement", 0.0)),
            }
        # dict でなければ override なし扱い
        return None

    # ---------------------------
    # 2) 通常（auto）モード
    # ---------------------------
    if emotion_ai is None:
        return None

    short: Optional[EmotionResult] = getattr(
        emotion_ai, "last_short_result", None
    )
    if short is None:
        return None

    return {
        "mode": short.mode or "normal",
        "affection": float(short.affection),
        "arousal": float(short.arousal),
        "tension": float(short.tension),
        "anger": float(short.anger),
        "sadness": float(short.sadness),
        "excitement": float(short.excitement),
    }
