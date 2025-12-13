# actors/mixer_ai.py
from __future__ import annotations
from typing import Any, Dict
import streamlit as st


class MixerAI:
    def __init__(self, *, state, emotion_ai, scene_ai) -> None:
        self.state = state
        self.emotion_ai = emotion_ai
        self.scene_ai = scene_ai

    def build_emotion_override(self) -> Dict[str, Any]:
        """
        manual_controls が未初期化／途中状態でも
        絶対に落ちない Mixer
        """
        manual = self.state.get("emotion_manual_controls") or {}
        world_manual = self.state.get("world_state_manual_controls") or {}

        # ★ dict 以外は無効化
        if not isinstance(manual, dict):
            manual = {}
        if not isinstance(world_manual, dict):
            world_manual = {}

        try:
            payload = {
                "emotion": {
                    "doki_power": float(manual.get("doki_power", 0.0)),
                    "masking_level": int(manual.get("masking_level", 0)),
                    "relationship_level": int(manual.get("relationship_level", 0)),
                },
                "world_state": {
                    "others_present": bool(world_manual.get("others_present", False)),
                },
                "scene_emotion": {},
            }
        except Exception:
            payload = {}

        return payload
