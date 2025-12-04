# actors/scene_ai.py
from __future__ import annotations

from typing import Any, Dict, Optional, Mapping
import os
import copy

import streamlit as st

from actors.scene.scene_manager import SceneManager


# Lyra 全体で共有する「世界のデフォルト」
DEFAULT_WORLD: Dict[str, Any] = {
    "time": {
        "slot": "morning",   # morning / lunch / after_school / night ...
        "time_str": "07:30", # 表示用の任意文字列
    },
    "weather": "clear",
    "locations": {
        "player": "プレイヤーの部屋",
        "floria": "プレイヤーの部屋",
    },
}


class SceneAI:
    """
    世界状態（world_state）と、SceneManager 由来の感情ボーナスを扱う AI。

    - world_state の正本は llm_meta["world"] に置く。
    - SessionState にも互換キーとしてミラーする：
        - scene_location   : プレイヤー位置
        - scene_time_slot  : 時間帯スロット
        - scene_time_str   : 時刻 "HH:MM"
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

        # llm_meta を取得（なければ空 dict）
        llm_meta = self.state.get("llm_meta")
        if not isinstance(llm_meta, dict):
            llm_meta = {}
        self.llm_meta: Dict[str, Any] = llm_meta

        # world_state を初期化（なければ DEFAULT_WORLD を採用）
        self.world: Dict[str, Any] = self.ensure_world_initialized()

        # SceneManager はセッションで共有
        key = "scene_manager"
        if key not in self.state:
            mgr = SceneManager(
                path="actors/scene/scene_bonus/scene_emotion_map.json"
            )
            mgr.load()
            self.state[key] = mgr
        self.manager: SceneManager = self.state[key]

    # ==================================================
    # 世界状態の初期化・同期
    # ==================================================
    def ensure_world_initialized(self) -> Dict[str, Any]:
        """
        self.llm_meta / self.state に world_state がなければ DEFAULT_WORLD をセットし、
        必要なキーを埋めてから戻す。
        """
        world = self.llm_meta.get("world")
        if not isinstance(world, dict):
            world = {}

        # デフォルトを埋める
        base = copy.deepcopy(DEFAULT_WORLD)

        # time
        t = world.get("time") or {}
        time_slot = t.get("slot") or base["time"]["slot"]
        time_str = t.get("time_str") or base["time"]["time_str"]

        # locations
        locs = world.get("locations") or {}
        player_loc = locs.get("player") or base["locations"]["player"]
        floria_loc = locs.get("floria") or base["locations"]["floria"]

        # weather
        weather = world.get("weather") or base["weather"]

        world = {
            "time": {
                "slot": time_slot,
                "time_str": time_str,
            },
            "weather": weather,
            "locations": {
                "player": player_loc,
                "floria": floria_loc,
            },
        }

        self.llm_meta["world"] = world
        self.state["llm_meta"] = self.llm_meta

        # 旧キーとの互換ミラー（SceneManager / CouncilManager などが参照）
        self.state["scene_location"] = player_loc
        self.state["scene_time_slot"] = time_slot
        self.state["scene_time_str"] = time_str

        return world

    def save_world(self) -> None:
        """
        self.world の内容を llm_meta / state に書き戻す。
        """
        # 一応最低限のキーを保証
        self.world.setdefault("time", {})
        self.world.setdefault("locations", {})
        self.world.setdefault("weather", DEFAULT_WORLD["weather"])

        t = self.world["time"]
        locs = self.world["locations"]

        t.setdefault("slot", DEFAULT_WORLD["time"]["slot"])
        t.setdefault("time_str", DEFAULT_WORLD["time"]["time_str"])
        locs.setdefault("player", DEFAULT_WORLD["locations"]["player"])
        locs.setdefault("floria", DEFAULT_WORLD["locations"]["floria"])

        self.llm_meta["world"] = self.world
        self.state["llm_meta"] = self.llm_meta

        self.state["scene_location"] = locs["player"]
        self.state["scene_time_slot"] = t["slot"]
        self.state["scene_time_str"] = t["time_str"]

    # ==================================================
    # public API
    # ==================================================
    def get_world_state(self) -> Dict[str, Any]:
        """
        現在の world_state 全体を返す。
        形式例：
        {
          "time": {"slot": "morning", "time_str": "07:30"},
          "weather": "clear",
          "locations": {"player": "駅前", "floria": "学食"},
        }
        """
        # 念のため毎回最低限のキーを保証
        self.world = self.ensure_world_initialized()
        return copy.deepcopy(self.world)

    def move_player(
        self,
        location: str,
        *,
        time_slot: Optional[str] = None,
        time_str: Optional[str] = None,
    ) -> None:
        """
        プレイヤーの位置（＋必要なら時間帯）を更新する。
        フローリアはここでは動かさない（将来別メソッドで自由移動）。
        """
        self.world = self.ensure_world_initialized()
        locs = self.world.setdefault("locations", {})
        locs["player"] = location

        if time_slot is not None:
            self.world.setdefault("time", {})["slot"] = time_slot
        if time_str is not None:
            self.world.setdefault("time", {})["time_str"] = time_str

        self.save_world()

    def move_floria(
        self,
        location: str,
    ) -> None:
        """
        フローリアだけを移動させる（時間帯は変更しない）。
        """
        self.world = self.ensure_world_initialized()
        locs = self.world.setdefault("locations", {})
        locs["floria"] = location
        self.save_world()

    # --------------------------------------------------
    # SceneManager から感情ボーナスを取得
    # --------------------------------------------------
    def get_scene_emotion(
        self,
        *,
        for_who: str = "player",
    ) -> Dict[str, float]:
        """
        SceneManager から感情ボーナスベクトルを取得する。
        今のところ「どの場所の感情を使うか」は for_who で指定。
        """
        self.world = self.ensure_world_initialized()

        locs = self.world.get("locations", {})
        t = self.world.get("time", {})

        if for_who == "floria":
            location = locs.get("floria") or locs.get("player") or DEFAULT_WORLD["locations"]["player"]
        else:
            location = locs.get("player") or DEFAULT_WORLD["locations"]["player"]

        slot_name = t.get("slot")
        time_str = t.get("time_str")

        return self.manager.get_for(
            location=location,
            time_str=time_str,
            slot_name=slot_name,
        )

    def get_emotion_bonus(self) -> Dict[str, float]:
        """
        MixerAI から呼び出される想定のラッパ。
        現時点では「プレイヤーの位置」に基づくボーナスを返す。
        """
        return self.get_scene_emotion(for_who="player")

    # --------------------------------------------------
    # MixerAI などに渡しやすいまとめペイロード
    # --------------------------------------------------
    def build_emotion_override_payload(self) -> Dict[str, Any]:
        """
        world_state + scene_emotion（プレイヤー基準）をまとめて返す。
        """
        ws = self.get_world_state()
        emo = self.get_scene_emotion(for_who="player")

        return {
            "world_state": ws,
            "scene_emotion": emo,
        }
