# actors/mixer_ai.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import os
import streamlit as st

from actors.emotion_ai import EmotionAI
from actors.scene_ai import SceneAI
from actors.utils.debug_world import debug_world_state  # さっき作ったやつを想定

LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"


@dataclass
class MixerAI:
    """
    EmotionAI / SceneAI / 手動スライダーを束ねて
    AnswerTalker に渡す emotion_override を構築するクラス。

    - SceneAI から world_state / scene_emotion を取得
    - DokiPowerControl 由来の emotion_manual_controls /
      world_state_manual_controls をマージ
    - 最終的な world_state に others_present を必ず反映する
    """

    state: Mapping[str, Any]
    emotion_ai: EmotionAI
    scene_ai: SceneAI

    def __init__(
        self,
        *,
        state: Optional[Mapping[str, Any]] = None,
        emotion_ai: EmotionAI,
        scene_ai: SceneAI,
    ) -> None:
        env_debug = os.getenv("LYRA_DEBUG", "")

        if state is not None:
            self.state = state
        elif env_debug == "1":
            self.state = st.session_state
        else:
            self.state = st.session_state

        self.emotion_ai = emotion_ai
        self.scene_ai = scene_ai

    # ----------------------------------------------------------
    # 内部ヘルパ
    # ----------------------------------------------------------
    def _get_manual_emotion(self) -> Dict[str, Any]:
        manual = self.state.get("emotion_manual_controls")
        if isinstance(manual, dict):
            return dict(manual)
        return {}

    def _get_manual_world(self) -> Dict[str, Any]:
        manual = self.state.get("world_state_manual_controls")
        if isinstance(manual, dict):
            return dict(manual)
        return {}

    @staticmethod
    def _calc_doki_level_from_power(doki_power: float) -> int:
        if doki_power >= 85:
            return 4
        if doki_power >= 60:
            return 3
        if doki_power >= 40:
            return 2
        if doki_power >= 20:
            return 1
        return 0

    # ----------------------------------------------------------
    # メイン：emotion_override 構築
    # ----------------------------------------------------------
    def build_emotion_override(self) -> Dict[str, Any]:
        """
        AnswerTalker / PersonaBase へ渡す emotion_override を組み立てる。

        戻り値の構造:
        {
          "world_state": {...},   # others_present を含む
          "scene_emotion": {...},
          "emotion": {...},       # relationship_level / masking_degree など
        }
        """

        # 1) SceneAI から world_state / scene_emotion を取得
        scene_payload = self.scene_ai.build_emotion_override_payload()
        world_state = scene_payload.get("world_state") or {}
        if not isinstance(world_state, dict):
            world_state = {}
        scene_emotion = scene_payload.get("scene_emotion") or {}
        if not isinstance(scene_emotion, dict):
            scene_emotion = {}

        # 2) 手動スライダー情報
        emo_manual = self._get_manual_emotion()
        ws_manual = self._get_manual_world()

        relationship_level = float(emo_manual.get("relationship_level", 0.0) or 0.0)
        doki_power = float(emo_manual.get("doki_power", 0.0) or 0.0)
        masking_level = float(emo_manual.get("masking_level", 0.0) or 0.0)
        environment = emo_manual.get("environment")  # "alone" / "with_others" or None

        doki_level = int(
            emo_manual.get("doki_level", self._calc_doki_level_from_power(doki_power))
            or 0
        )
        masking_degree = max(0.0, min(masking_level / 100.0, 1.0))

        # 3) others_present を決定
        others_present: Optional[bool] = None

        # 3-1) world_state_manual_controls が最優先
        if isinstance(ws_manual.get("others_present"), bool):
            others_present = ws_manual["others_present"]

        # 3-2) なければ environment から推定
        if others_present is None and isinstance(environment, str):
            if environment == "alone":
                others_present = False
            elif environment == "with_others":
                others_present = True

        # 3-3) それでも None なら元の world_state を尊重
        if others_present is None:
            raw = world_state.get("others_present")
            if isinstance(raw, bool):
                others_present = raw

        # 3-4) 決まったら world_state に書き込む
        if isinstance(others_present, bool):
            world_state["others_present"] = others_present

        # 4) emotion ブロックを組み立て
        #    （affection 系は SceneEmotion 側の補正があればそれを使う）
        base_affection = float(scene_emotion.get("affection", 0.0) or 0.0)
        base_arousal = float(scene_emotion.get("arousal", 0.0) or 0.0)

        emotion: Dict[str, Any] = {
            "mode": emo_manual.get("mode", "normal"),
            "affection": base_affection,
            "arousal": base_arousal,
            "doki_power": doki_power,
            "doki_level": doki_level,
            "relationship_level": relationship_level,
            # 互換用フィールド（古いコードが読むかもしれないので残す）
            "relationship": relationship_level,
            "masking_degree": masking_degree,
            "masking": masking_degree,
        }

        # 5) デバッグ出力
        debug_world_state(
            caller="MixerAI.build_emotion_override",
            step="after_merge",
            world_state=world_state,
            scene_emotion=scene_emotion,
            emotion=emotion,
            extra={
                "has_emo_manual": bool(emo_manual),
                "has_ws_manual": bool(ws_manual),
            },
        )

        # 6) 結果を返す
        return {
            "world_state": world_state,
            "scene_emotion": scene_emotion,
            "emotion": emotion,
        }
