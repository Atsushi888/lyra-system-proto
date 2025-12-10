# actors/mixer_ai.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import os

import streamlit as st

from actors.emotion_ai import EmotionAI
from actors.scene_ai import SceneAI
from actors.utils.debug_world_state import WorldStateDebugger


LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"

# 🔍 このファイル専用デバッガ
WS_DEBUGGER = WorldStateDebugger(name="MixerAI")


@dataclass
class MixerAI:
    """
    EmotionAI / SceneAI / 手動パラメータを混ぜて
    AnswerTalker に渡す emotion_override を組み立てる担当。

    ※ world_state 自体は SceneAI が正規窓口。
    """

    state: Mapping[str, Any]
    emotion_ai: EmotionAI
    scene_ai: SceneAI

    def __init__(
        self,
        *,
        state: Optional[Mapping[str, Any]] = None,
        emotion_ai: Optional[EmotionAI] = None,
        scene_ai: Optional[SceneAI] = None,
    ) -> None:
        if state is not None:
            self.state = state
        else:
            self.state = st.session_state

        if emotion_ai is None:
            raise ValueError("MixerAI: emotion_ai が None です。")
        if scene_ai is None:
            raise ValueError("MixerAI: scene_ai が None です。")

        self.emotion_ai = emotion_ai
        self.scene_ai = scene_ai

    # ======================================================
    # emotion_override の組み立て（AnswerTalker から呼ばれる）
    # ======================================================
    def build_emotion_override(self) -> Dict[str, Any]:
        """
        AnswerTalker → PersonaBase.build_emotion_based_system_prompt_core に渡す
        emotion_override を構築する。

        返却フォーマット:
        {
            "world_state": {...},
            "scene_emotion": {...},
            "emotion": {...},  # affection / doki / relationship / masking など
        }
        """

        # 1) SceneAI から world_state / scene_emotion を取得
        world_state = self.scene_ai.get_world_state()
        scene_emotion = self.scene_ai.get_scene_emotion(world_state)

        # 2) llm_meta から EmotionAI 関連の状態を拾う（あくまで読み取り only）
        llm_meta = self.state.get("llm_meta") or {}
        if not isinstance(llm_meta, dict):
            llm_meta = {}

        emotion_short = llm_meta.get("emotion") or {}
        if not isinstance(emotion_short, dict):
            emotion_short = {}

        emotion_long = llm_meta.get("emotion_long_term") or {}
        if not isinstance(emotion_long, dict):
            emotion_long = {}

        # 3) doki_power / affection_with_doki などを合成（ここは元のロジックを使う想定）
        #    ↓↓↓ ★ ここから先は「あなたの元コード」をそのまま貼り付けてOK ↓↓↓

        # ベースの affection（短期）があれば優先、なければ長期から拾う
        affection = float(
            emotion_short.get("affection_with_doki", emotion_short.get("affection", 0.0))
            or emotion_long.get("affection_with_doki", emotion_long.get("affection", 0.0))
            or 0.0
        )

        doki_power = float(
            emotion_short.get("doki_power", emotion_long.get("doki_power", 0.0)) or 0.0
        )
        doki_level = int(
            emotion_short.get("doki_level", emotion_long.get("doki_level", 0)) or 0
        )

        relationship_level = float(
            emotion_short.get("relationship_level", emotion_long.get("relationship_level", 0.0))
            or 0.0
        )
        relationship_stage = (
            emotion_short.get("relationship_stage")
            or emotion_long.get("relationship_stage")
            or ""
        )

        masking_degree = float(
            emotion_short.get("masking_degree", emotion_long.get("masking_degree", 0.0))
            or 0.0
        )

        # affection_zone は EmotionAI 側で決めたものがあればそれを尊重
        affection_zone = (
            emotion_short.get("affection_zone")
            or emotion_long.get("affection_zone")
            or "auto"
        )

        # 4) override payload を組み立て
        emotion_payload: Dict[str, Any] = {
            "affection": affection,
            "affection_with_doki": affection,
            "affection_zone": affection_zone,
            "doki_power": doki_power,
            "doki_level": doki_level,
            "relationship_level": relationship_level,
            "relationship_stage": relationship_stage,
            "masking_degree": masking_degree,
        }

        emotion_override: Dict[str, Any] = {
            "world_state": world_state,
            "scene_emotion": scene_emotion,
            "emotion": emotion_payload,
        }

        # 5) 🔍 デバッグ：MixerAI 時点の world_state / emotion_override を丸ごとダンプ
        WS_DEBUGGER.log(
            caller="MixerAI.build_emotion_override",
            world_state=world_state,
            scene_emotion=scene_emotion,
            emotion=emotion_payload,
            extra={
                "has_llm_meta": bool(llm_meta),
                "has_emotion_short": bool(emotion_short),
                "has_emotion_long": bool(emotion_long),
            },
        )

        # 6) そのまま AnswerTalker へ返す
        return emotion_override
