# actors/mixer_ai.py
from __future__ import annotations
from typing import Any, Dict, Optional, Mapping

class MixerAI:
    def __init__(self, state: Mapping[str, Any], emotion_ai, scene_ai) -> None:
        self.state = state
        self.emotion_ai = emotion_ai
        self.scene_ai = scene_ai

    # ---------------------------------------------------------
    # Emotion / WorldState / Manual override の統合
    # ---------------------------------------------------------
    def build_emotion_override(self) -> Dict[str, Any]:

        # SceneAI が持つ world_state / scene_emotion を基礎にする
        ws = self.scene_ai.get_world_state()
        scene_emotion = self.scene_ai.get_scene_emotion(ws)

        # 直近ターンの EmotionResult（llm_meta にあるやつ）
        llm_meta = self.state.get("llm_meta") or {}
        base_emotion = llm_meta.get("emotion") or {}

        # DokipowerControl からの手動オーバーライド
        emo_manual = self.state.get("emotion_manual_controls") or {}
        ws_manual = self.state.get("world_state_manual_controls") or {}

        # ---- EmotionResult の手動上書き ----
        emotion: Dict[str, Any] = dict(base_emotion)
        for k, v in emo_manual.items():
            emotion[k] = v

        # environment → others_around 連携（従来仕様）
        env = emo_manual.get("environment")
        if env == "alone":
            ws["others_around"] = False
        elif env == "with_others":
            ws["others_around"] = True

        # ---------------------------------------------------------
        # ★ world_state_manual_controls["others_present"] を
        #    必ず “厳密な bool 値” に変換して world_state に反映
        # ---------------------------------------------------------
        if "others_present" in ws_manual:
            raw = ws_manual["others_present"]

            # 文字列なら true/false を正規化
            if isinstance(raw, str):
                sval = raw.strip().lower()
                val = sval in ("1", "true", "yes", "on")
            else:
                val = bool(raw)

            ws["others_present"] = val

            # Emotion 側にも反映させる場合
            emotion["others_present"] = val

        # ---------------------------------------------------------
        # 統合 payload
        # ---------------------------------------------------------
        return {
            "world_state": ws,
            "scene_emotion": scene_emotion,
            "emotion": emotion,
        }
