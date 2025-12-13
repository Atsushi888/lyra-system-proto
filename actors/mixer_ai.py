from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional
import os

import streamlit as st

from actors.scene_ai import SceneAI
from actors.emotion_ai import EmotionAI


LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"


def _dwrite(*args: Any) -> None:
    if LYRA_DEBUG:
        st.write(*args)


class MixerAI:
    """
    SceneAI の推定 + 手動デバッグ（DokipowerControl 等）を合成して
    ModelsAI に渡す emotion_override payload を作る。

    返却形式（最低限）:
    {
      "emotion": {...},        # relationship_level / doki_power / masking_level 等
      "world_state": {...},    # others_present 等
      "scene_emotion": {...},  # SceneAI の推定
    }
    """

    def __init__(
        self,
        *,
        state: MutableMapping[str, Any],
        emotion_ai: Optional[EmotionAI] = None,
        scene_ai: Optional[SceneAI] = None,
    ) -> None:
        self.state = state
        self.emotion_ai = emotion_ai
        self.scene_ai = scene_ai or SceneAI(state=self.state)

    def _get_manual_emotion(self) -> Dict[str, Any]:
        manual = self.state.get("emotion_manual_controls")
        if isinstance(manual, dict):
            # models へ渡すのは “必要キーだけ”
            out: Dict[str, Any] = {}
            if "relationship_level" in manual:
                out["relationship_level"] = int(manual["relationship_level"])
            if "doki_power" in manual:
                out["doki_power"] = float(manual["doki_power"])
            if "masking_level" in manual:
                out["masking_level"] = int(manual["masking_level"])
            # interaction_mode_hint は world_state 側で使う想定
            return out
        return {}

    def _get_manual_world(self) -> Dict[str, Any]:
        ws_manual = self.state.get("world_state_manual_controls")
        if isinstance(ws_manual, dict):
            out: Dict[str, Any] = {}
            if "others_present" in ws_manual:
                out["others_present"] = bool(ws_manual["others_present"])
            if "interaction_mode_hint" in ws_manual:
                out["interaction_mode_hint"] = str(ws_manual["interaction_mode_hint"])
            return out

        # 互換：emotion_manual_controls から拾える場合
        emo_manual = self.state.get("emotion_manual_controls")
        if isinstance(emo_manual, dict):
            out2: Dict[str, Any] = {}
            if "others_present" in emo_manual:
                out2["others_present"] = bool(emo_manual["others_present"])
            if "interaction_mode_hint" in emo_manual:
                out2["interaction_mode_hint"] = str(emo_manual["interaction_mode_hint"])
            return out2

        return {}

    def build_emotion_override(self) -> Dict[str, Any]:
        # SceneAI 推定
        scene_payload: Dict[str, Any] = {}
        try:
            scene_payload = self.scene_ai.build_emotion_override_payload()
        except Exception as e:
            if LYRA_DEBUG:
                st.exception(e)
            scene_payload = {"world_state": {}, "scene_emotion": {}}

        scene_world = scene_payload.get("world_state") or {}
        scene_emotion = scene_payload.get("scene_emotion") or {}

        # 手動上書き
        manual_emotion = self._get_manual_emotion()
        manual_world = self._get_manual_world()

        # world_state は “SceneAI を基礎” にして manual で上書き
        merged_world: Dict[str, Any] = dict(scene_world) if isinstance(scene_world, dict) else {}
        if isinstance(manual_world, dict):
            merged_world.update(manual_world)

        payload = {
            "emotion": manual_emotion,                 # いまは手動値のみ（必要なら Scene 感情と合成しても良い）
            "world_state": merged_world,
            "scene_emotion": scene_emotion if isinstance(scene_emotion, dict) else {},
        }

        _dwrite("[DEBUG:MixerAI] build_emotion_override payload=", payload)
        return payload
