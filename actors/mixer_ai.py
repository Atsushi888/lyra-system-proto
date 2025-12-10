# actors/mixer_ai.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional
import os

import streamlit as st

from actors.emotion_ai import EmotionAI
from actors.scene_ai import SceneAI
from actors.utils.debug_world_state import WorldStateDebugger


@dataclass
class MixerAI:
    """
    EmotionAI / SceneAI / world_state を束ねて
    PersonaBase に渡す emotion_override を組み立てるクラス。

    - world_state: SceneAI / DokiPowerControl が管理している値を「極力そのまま」通す
      （特に `others_present` を勝手に潰さないことが今回の肝）
    - scene_emotion: SceneManager 由来のシーン補正
    - emotion: 直近ターンの EmotionResult を llm_meta から引き継ぐ（あれば）

    ※ デバッグ時は WorldStateDebugger で world_state を丸ごと吐き出す。
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
            # デバッグ中は session_state を共有
            self.state = st.session_state
        else:
            # 現状 Streamlit 前提
            self.state = st.session_state

        self.emotion_ai = emotion_ai
        self.scene_ai = scene_ai

        # 共通デバッガ
        self._ws_debugger = WorldStateDebugger(name="MixerAI")

    # ==========================================================
    # 内部ヘルパ
    # ==========================================================
    def _get_llm_meta(self) -> Dict[str, Any]:
        meta = self.state.get("llm_meta")
        if not isinstance(meta, dict):
            meta = {}
        return meta

    # ==========================================================
    # 公開 API
    # ==========================================================
    def build_emotion_override(self) -> Dict[str, Any]:
        """
        AnswerTalker から呼ばれる想定のメインメソッド。

        以下の形の dict を返す：
        {
            "world_state": {...},
            "scene_emotion": {...},
            "emotion": {...},
        }

        PersonaBase.build_emotion_based_system_prompt() 側では
        これをそのまま受け取り、world_state / scene_emotion / emotion に分解している。
        """

        # ---- world_state を取得 ----
        # DokiPowerControl が触った world_state が state に入っている前提。
        ws_raw = self.state.get("world_state")
        if not isinstance(ws_raw, dict) or not ws_raw:
            # まだ何も無い場合のみ SceneAI に初期化させる
            world_state = self.scene_ai.get_world_state()
        else:
            # 既存 world_state を尊重しつつ、party.mode 等だけ SceneAI で整合させる。
            # ※ set_world_state() は others_present など余計なキーを消さない実装なので、
            #    DokiPowerControl 側で立てたフラグはそのまま残る。
            self.scene_ai.set_world_state(ws_raw)
            world_state = self.scene_ai.get_world_state()

        # ---- scene_emotion を算出 ----
        scene_emotion = self.scene_ai.get_scene_emotion(world_state)

        # ---- Emotion 情報（あれば）を引き継ぎ ----
        llm_meta = self._get_llm_meta()
        emotion_meta = llm_meta.get("emotion") or {}

        # Mixer が返す統合 payload
        emotion_override: Dict[str, Any] = {
            "world_state": world_state,
            "scene_emotion": scene_emotion,
            "emotion": emotion_meta,
        }

        # ---- デバッグ出力 ----
        # LYRA_DEBUG=1 のときだけ、world_state＆emotion_override を丸ごと吐く
        self._ws_debugger.log(
            caller="MixerAI.build_emotion_override",
            world_state=world_state,
            emotion_override=emotion_override,
            extra={"has_emotion_meta": bool(emotion_meta)},
        )

        return emotion_override
