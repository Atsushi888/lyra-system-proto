# actors/scene_ai.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import streamlit as st


@dataclass
class SceneAI:
    """
    Lyra 全体で共有する「世界のデフォルト状態」を管理するクラス。

    - world_state は Streamlit の session_state["world_state"] に保持
    - player / floria の位置は完全に独立して管理する
    - プレイヤー移動とフローリア移動は別メソッド
    """

    state: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        # 明示的 state があればそれを、なければ st.session_state を使う
        if self.state is None:
            self.state = st.session_state
        # world_state のデフォルトを確保
        self._ensure_world_defaults()

    # =========================================================
    # 基本 world_state アクセス
    # =========================================================
    def _ensure_world_defaults(self) -> Dict[str, Any]:
        """world_state の最低限のキーを必ず用意する。"""
        world = self.state.get("world_state")
        if not isinstance(world, dict):
            world = {}

        # 場所情報
        locs = world.get("locations")
        if not isinstance(locs, dict):
            locs = {}

        # デフォルトは「プレイヤーの部屋」で両者スタート
        if "player" not in locs:
            locs["player"] = "プレイヤーの部屋"
        if "floria" not in locs:
            locs["floria"] = "プレイヤーの部屋"

        world["locations"] = locs

        # 時刻情報
        t = world.get("time")
        if not isinstance(t, dict):
            t = {}

        t.setdefault("slot", "morning")
        t.setdefault("time_str", "07:30")

        world["time"] = t

        # 保存
        self.state["world_state"] = world
        return world

    def get_world_state(self) -> Dict[str, Any]:
        """常に正規化された world_state を返す。"""
        return self._ensure_world_defaults()

    def set_world_state(self, world: Dict[str, Any]) -> None:
        """world_state を直接上書きしたいとき用。"""
        if not isinstance(world, dict):
            world = {}
        self.state["world_state"] = world
        # 念のため整形
        self._ensure_world_defaults()

    # =========================================================
    # キャラクター移動
    # =========================================================
    def move_player(
        self,
        location: str,
        *,
        time_slot: Optional[str] = None,
        time_str: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        プレイヤーだけを移動させる。

        ※ ここでは絶対に floria の位置を変更しない。
        """
        world = self._ensure_world_defaults()
        locs = world["locations"]
        time_info = world["time"]

        locs["player"] = location  # ★ プレイヤーだけ更新

        if time_slot is not None:
            time_info["slot"] = time_slot
        if time_str is not None:
            time_info["time_str"] = time_str

        world["locations"] = locs
        world["time"] = time_info
        self.state["world_state"] = world
        return world

    def move_floria(self, location: str) -> Dict[str, Any]:
        """
        フローリアだけを移動させる。

        時刻・時間帯はプレイヤーと共有のまま。
        """
        world = self._ensure_world_defaults()
        locs = world["locations"]

        locs["floria"] = location  # ★ フローリアだけ更新

        world["locations"] = locs
        self.state["world_state"] = world
        return world

    # =========================================================
    # LLM へのコンテキスト注入用ユーティリティ
    # =========================================================
    def build_emotion_override_payload(self) -> Dict[str, Any]:
        """
        AnswerTalker から呼ばれ、world_state と scene_emotion をまとめて返す。

        scene_emotion は「プレイヤーの現在位置＆時刻」に基づく感情補正ベクトル。
        """
        world = self._ensure_world_defaults()
        locs = world["locations"]
        t = world["time"]

        player_loc = locs.get("player", "プレイヤーの部屋")
        time_slot = t.get("slot", "morning")
        time_str = t.get("time_str", "07:30")

        # 遅延 import で循環参照を回避
        from actors.scene.scene_manager import SceneManager

        mgr = SceneManager()
        mgr.load()
        emo_vec = mgr.get_for(
            location=player_loc,
            time_str=time_str,
            slot_name=time_slot,
        )

        scene_emotion = {
            "location": player_loc,
            "time_slot": time_slot,
            "time_str": time_str,
            "vector": emo_vec,
        }

        # llm_meta にもコピーしておく（あれば）
        llm_meta = self.state.get("llm_meta")
        if isinstance(llm_meta, dict):
            llm_meta["world_state"] = world
            llm_meta["scene_emotion"] = scene_emotion
            self.state["llm_meta"] = llm_meta

        return {
            "world_state": world,
            "scene_emotion": scene_emotion,
        }

    def get_prompt_world_state_text(self) -> str:
        """
        Council / Narrator 用に、人間が読める形で world_state をテキスト化。
        """
        world = self._ensure_world_defaults()
        locs = world["locations"]
        t = world["time"]

        return (
            "【現在の状況メモ】\n"
            f"- プレイヤー位置: {locs.get('player')}\n"
            f"- フローリア位置: {locs.get('floria')}\n"
            f"- 時間帯: {t.get('slot')} / 時刻: {t.get('time_str')}\n"
        )
