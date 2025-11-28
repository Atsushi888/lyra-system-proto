# components/emotion_control.py
from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from actors.emotion_ai import EmotionResult

# session_state のキーを定数として定義
EMOTION_OVERRIDE_MODE_KEY = "emotion_override_mode"     # "auto" / "manual_full"
EMOTION_OVERRIDE_MANUAL_KEY = "emotion_override_manual"  # Dict[str, Any] を想定


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
            # manual dict 側で key が足りなくても安全に読む
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
