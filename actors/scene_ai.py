# actors/scene_ai.py
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
        env_debug = os.getenv("LYRA_DEBUG", "")

        if state is not None:
            self.state = state
        elif env_debug == "1":
            self.state = st.session_state
        else:
            self.state = st.session_state

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

        - scene_location / scene_time_slot / scene_time_str が未設定なら、
          SceneManager の情報からデフォルト値を決めて state に書き戻す。
        """
        # 場所
        location = self.state.get("scene_location")
        loc_names = list(self.manager.locations.keys())
        if not location:
            if "プレイヤーの部屋" in self.manager.locations:
                location = "プレイヤーの部屋"
            elif loc_names:
                location = loc_names[0]
            else:
                location = "通学路"
            self.state["scene_location"] = location

        # 時間帯スロット
        slot_name = self.state.get("scene_time_slot")
        slot_keys = list(self.manager.time_slots.keys())
        if not slot_name:
            if "morning" in self.manager.time_slots:
                slot_name = "morning"
            elif slot_keys:
                slot_name = slot_keys[0]
            else:
                slot_name = None
            self.state["scene_time_slot"] = slot_name

        # 時刻文字列
        time_str = self.state.get("scene_time_str")
        if not time_str:
            default_time = "07:30"
            if slot_name and slot_name in self.manager.time_slots:
                default_time = self.manager.time_slots[slot_name].get("start", default_time)
            time_str = default_time
            self.state["scene_time_str"] = time_str

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
        """
        world_state をもとに SceneManager から感情補正ベクトルを取得する。
        """
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

    # -----------------------------
    # MixerAI 向けの簡易 API
    # -----------------------------
    def build_emotion_override_payload(self) -> Dict[str, Any]:
        ws = self.get_world_state()
        emo = self.get_scene_emotion(ws)

        return {
            "world_state": ws,
            "scene_emotion": emo,
        }

    # 旧 MixerAI 互換用（Scene ボーナスだけ返す）
    def get_emotion_bonus(self) -> Dict[str, float]:
        return self.get_scene_emotion()
