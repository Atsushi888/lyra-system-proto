from __future__ import annotations

from typing import Any, Dict, Optional, Mapping
import os

import streamlit as st

from actors.scene.scene_manager import SceneManager


class SceneAI:
    """
    シーン情報（場所・時間帯）から
    SceneManager 経由で感情補正ベクトルを取り出す役。

    - state: Streamlit の session_state か、外部から渡された dict 互換オブジェクト
    - SceneManager は state["scene_manager"] に共有して使う
    """

    def __init__(self, state: Optional[Mapping[str, Any]] = None) -> None:
        # AnswerTalker と同じパターンで state を決める
        env_debug = os.getenv("LYRA_DEBUG", "")

        if state is not None:
            self.state = state
        elif env_debug == "1":
            self.state = st.session_state
        else:
            self.state = st.session_state

        # SceneManager をセッション内で 1個だけ確保
        key = "scene_manager"
        if key not in self.state:
            mgr = SceneManager(
                path="actors/scene/scene_bonus/scene_emotion_map.json"
            )
            mgr.load()
            self.state[key] = mgr

        self.manager: SceneManager = self.state[key]

    # -----------------------------
    # world_state の取得
    # -----------------------------
    def get_world_state(self) -> Dict[str, Any]:
        """
        現在の world_state を返す。
        """
        location = self.state.get("scene_location", "通学路")
        slot_name = self.state.get("scene_time_slot", None)   # ex: "morning"
        time_str = self.state.get("scene_time_str", None)     # ex: "07:45"

        return {
            "location": location,
            "time_slot": slot_name,
            "time_str": time_str,
        }

    # -----------------------------
    # SceneManager から感情補正を取得
    # -----------------------------
    def get_scene_emotion(
        self,
        world_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        if world_state is None:
            world_state = self.get_world_state()

        location = world_state.get("location", "通学路")
        slot_name = world_state.get("time_slot")
        time_str = world_state.get("time_str")

        return self.manager.get_for(
            location=location,
            time_str=time_str,
            slot_name=slot_name,
        )

    # 旧 MixerAI 向けの互換メソッド
    def get_emotion_bonus(self) -> Dict[str, float]:
        """
        互換用：SceneManager からのボーナスベクトル。
        """
        ws = self.get_world_state()
        return self.get_scene_emotion(ws)

    # -----------------------------
    # MixerAI 向けの簡易 API（オプション）
    # -----------------------------
    def build_emotion_override_payload(self) -> Dict[str, Any]:
        ws = self.get_world_state()
        emo = self.get_scene_emotion(ws)

        return {
            "world_state": ws,
            "scene_emotion": emo,
        }
