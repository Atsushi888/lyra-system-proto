# actors/mixer_ai.py
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from actors.emotion_ai import EmotionResult


class MixerAI:
    """
    EmotionAI / SceneAI / 手動デバッグ（ドキドキ💓）の結果をマージして
    ModelsAI に渡す emotion_override を組み立てるクラス。

    優先順位:
      1. ドキドキ💓デバッグ（st.session_state["mixer_debug_emotion"]）
      2. 直近ターンの EmotionAI 結果（llm_meta["emotion"]）
      3. SceneAI 由来の world_state / scene_emotion など
    """

    def __init__(
        self,
        *,
        state: Mapping[str, Any],
        emotion_ai: Any,
        scene_ai: Any,
    ) -> None:
        self.state = state
        self.emotion_ai = emotion_ai
        self.scene_ai = scene_ai

    # -------------------------------
    # 内部ヘルパ
    # -------------------------------
    def _get_debug_emotion(self) -> Optional[Dict[str, Any]]:
        """
        ドキドキ💓パワー調整画面からの手動オーバーライド。

        components/dokipower_control.py で
        st.session_state["mixer_debug_emotion"] に保存された dict をそのまま返す。
        """
        raw = self.state.get("mixer_debug_emotion")
        if isinstance(raw, dict):
            return raw
        return None

    def _get_last_emotion(self) -> Optional[Dict[str, Any]]:
        """
        直近ターンの EmotionAI.analyze() 結果を llm_meta から取得。
        """
        llm_meta = self.state.get("llm_meta") or {}
        emo = llm_meta.get("emotion")
        if isinstance(emo, dict):
            return emo
        return None

    # -------------------------------
    # public API
    # -------------------------------
    def build_emotion_override(self) -> Dict[str, Any]:
        """
        ModelsAI.collect() に渡す emotion_override を構築して返す。

        返り値の例:
        {
            "emotion_source": "debug_dokipower" | "auto",
            "emotion": {... EmotionResult dict ...},
            "world_state": {...},
            "scene_emotion": {...},
            "scene_error": "...",  # エラー時のみ
        }
        """
        override: Dict[str, Any] = {}

        # 3) SceneAI 側（ワールド・シーン情報）
        try:
            scene_payload = self.scene_ai.build_emotion_override_payload()
            if isinstance(scene_payload, dict):
                # world_state / scene_emotion などをそのまま詰める前提
                for k, v in scene_payload.items():
                    override[k] = v
        except Exception as e:
            override["scene_error"] = str(e)

        # 2) EmotionAI の直近結果（自動）
        last_emo = self._get_last_emotion()
        if last_emo:
            override.setdefault("emotion", last_emo)

        # 1) ドキドキ💓デバッグ（最優先）
        debug_emo = self._get_debug_emotion()
        if debug_emo:
            override["emotion"] = debug_emo
            override["emotion_source"] = "debug_dokipower"
        else:
            override.setdefault("emotion_source", "auto")

        return override
