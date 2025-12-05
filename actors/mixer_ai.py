# actors/mixer_ai.py
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import streamlit as st

from actors.emotion_ai import EmotionResult


class MixerAI:
    """
    SceneAI や ドキドキ💓デバッグ用 EmotionResult など、
    複数ソースの「感情情報」をまとめて扱うクラス。

    現段階では主に:

    - ドキドキ💓パワー・サイドウインドウからの手動 EmotionResult
      (`session_state["mixer_debug_emotion"]`)
    を拾って、AnswerTalker / ModelsAI に渡す `emotion_override` を組み立てる。
    """

    def __init__(
        self,
        *,
        state: Optional[Mapping[str, Any]] = None,
        emotion_ai: Optional[Any] = None,
        scene_ai: Optional[Any] = None,
    ) -> None:
        # Streamlit の session_state を共有
        self.state: Mapping[str, Any] = state or st.session_state
        self.emotion_ai = emotion_ai
        self.scene_ai = scene_ai

    # ---------------------------------------------------
    # 公開 API
    # ---------------------------------------------------
    def build_emotion_override(self) -> Dict[str, Any]:
        """
        ModelsAI.collect() に渡す emotion_override を組み立てる。

        返り値の例:

        {
            "enabled": true,
            "source": "dokipower_debug",
            "emotion": {
                "mode": "normal",
                "affection": 1.0,
                "arousal": 0.8,
                "tension": 0.1,
                "anger": 0.0,
                "sadness": 0.0,
                "excitement": 0.7,
                "raw_text": "(from dokipower_debug)",
                "doki_power": 100.0,
                "doki_level": 3,
                "meta": {},
            }
        }
        """
        override: Dict[str, Any] = {}

        # 1) ドキドキ💓パワー・サイドウインドウからのデバッグ EmotionResult
        debug_emo = self.state.get("mixer_debug_emotion")
        if isinstance(debug_emo, dict) and debug_emo:
            override["emotion"] = dict(debug_emo)
            override["source"] = "dokipower_debug"

        # 2) SceneAI 由来の情報は、llm_meta 側で直接扱うので
        #    ここでは override には含めない（必要になったら拡張）

        override["enabled"] = bool(override.get("emotion"))
        return override

    def set_manual_emotion(self, emo: EmotionResult) -> None:
        """
        外部から直接 EmotionResult を与えてデバッグしたいとき用のヘルパ。
        """
        self.state["mixer_debug_emotion"] = emo.to_dict()

    def clear_manual_emotion(self) -> None:
        """
        ドキドキ💓デバッグ用の手動感情をクリアする。
        """
        if "mixer_debug_emotion" in self.state:
            del self.state["mixer_debug_emotion"]
