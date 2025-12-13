# actors/mixer_ai.py
from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional
import os
import streamlit as st

from actors.scene_ai import SceneAI
from actors.emotion_ai import EmotionAI

LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"


def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


class MixerAI:
    """
    SceneAI / Emotion(手動) / WorldState(手動) を統合して
    ModelsAI2 に渡す emotion_override を構築する。
    """

    def __init__(
        self,
        *,
        state: MutableMapping[str, Any],
        emotion_ai: Optional[EmotionAI] = None,
        scene_ai: Optional[SceneAI] = None,
    ) -> None:
        self.state = state if state is not None else st.session_state
        self.emotion_ai = emotion_ai
        self.scene_ai = scene_ai or SceneAI(state=self.state)

    # -------------------------------------------------

    def build_emotion_override(self) -> Dict[str, Any]:
        """
        返却形式（基本）:
        {
          "emotion": { ... },
          "world_state": { ... },
          "scene_emotion": { ... }
        }
        """
        # 1) SceneAI 由来の payload
        try:
            scene_payload = self.scene_ai.build_emotion_override_payload()
        except Exception as e:
            scene_payload = {"world_state": {}, "scene_emotion": {}}
            if LYRA_DEBUG:
                st.error("[MixerAI] SceneAI payload failed")
                st.exception(e)

        world_state = _as_dict(scene_payload.get("world_state"))
        scene_emotion = _as_dict(scene_payload.get("scene_emotion"))

        # 2) 手動コントロール（dokipower_control の apply が書くやつ）
        manual_emo = _as_dict(self.state.get("emotion_manual_controls"))
        manual_ws = _as_dict(self.state.get("world_state_manual_controls"))

        # manual_emo から “emotion” に入れる値だけ抽出
        # （必要なキーが増えたらここへ追加）
        emotion_patch: Dict[str, Any] = {}
        if manual_emo:
            # relationship_level / doki_power / masking_level は既定
            if "relationship_level" in manual_emo:
                emotion_patch["relationship_level"] = manual_emo.get("relationship_level")
            if "doki_power" in manual_emo:
                emotion_patch["doki_power"] = manual_emo.get("doki_power")
            if "masking_level" in manual_emo:
                emotion_patch["masking_level"] = manual_emo.get("masking_level")

        # 3) world_state の手動上書き（others_present / interaction_mode_hint）
        if manual_emo.get("others_present") is not None:
            world_state["others_present"] = bool(manual_emo.get("others_present"))
        if manual_ws.get("others_present") is not None:
            world_state["others_present"] = bool(manual_ws.get("others_present"))

        # interaction_mode_hint は world_state に寄せる
        im_hint = manual_emo.get("interaction_mode_hint") or manual_ws.get("interaction_mode_hint")
        if im_hint:
            world_state["interaction_mode_hint"] = str(im_hint)

        # player_name を world_state に保険で入れる（ビューやプロンプトで使える）
        player_name = self.state.get("player_name") or "アツシ"
        if isinstance(player_name, str) and player_name:
            world_state.setdefault("player_name", player_name)

        # 4) 返却
        out = {
            "emotion": emotion_patch,
            "world_state": world_state,
            "scene_emotion": scene_emotion,
        }

        return out
