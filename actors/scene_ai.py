# actors/scene_ai.py
from __future__ import annotations

from typing import Any, Dict, Optional, Mapping
import os

import streamlit as st

from actors.scene.scene_manager import SceneManager


class SceneAI:
    """
    シーン情報（場所・時間帯）を管理し、
    - llm_meta["world_state"] に保存
    - SceneManager から感情ボーナスを取り出す
    - LLM 向けの「シーン説明 system プロンプト」を組み立てる
    役割を持つクラス。
    """

    def __init__(self, state: Optional[Mapping[str, Any]] = None) -> None:
        # AnswerTalker と同じパターンで state を決める
        env_debug = os.getenv("LYRA_DEBUG", "")

        if state is not None:
            self.state = state
        elif env_debug == "1":
            self.state = st.session_state
        else:
            # 現状は Streamlit 前提なので session_state を使う
            self.state = st.session_state

        # SceneManager（JSON 読み込み用）
        self.manager = SceneManager(
            path="actors/scene/scene_bonus/scene_emotion_map.json"
        )
        self.manager.load()

        # llm_meta を確保
        llm_meta = self.state.get("llm_meta")
        if not isinstance(llm_meta, dict):
            llm_meta = {}

        # world_state 初期化・補正
        ws_raw = llm_meta.get("world_state")
        if not isinstance(ws_raw, dict) or not ws_raw:
            ws = self._default_world_state()
        else:
            ws = self._merge_default_world_state(ws_raw)

        llm_meta["world_state"] = ws
        self.llm_meta: Dict[str, Any] = llm_meta
        self.state["llm_meta"] = self.llm_meta

    # ---------------------------------------------------------
    # world_state の基本構造
    # ---------------------------------------------------------
    @staticmethod
    def _default_world_state() -> Dict[str, Any]:
        """
        world_state のデフォルト。
        - 場所: プレイヤー/フローリアとも「プレイヤーの部屋」
        - 時間: morning / 07:30
        """
        return {
            "locations": {
                "player": "プレイヤーの部屋",
                "floria": "プレイヤーの部屋",
            },
            "time": {
                "slot": "morning",
                "time_str": "07:30",
            },
        }

    def _merge_default_world_state(self, src: Dict[str, Any]) -> Dict[str, Any]:
        """既存 world_state に足りないキーをデフォルトで埋める。"""
        base = self._default_world_state()

        loc_src = (src.get("locations") or {}) if isinstance(src, dict) else {}
        time_src = (src.get("time") or {}) if isinstance(src, dict) else {}

        locations = {
            "player": loc_src.get("player", base["locations"]["player"]),
            "floria": loc_src.get("floria", base["locations"]["floria"]),
        }
        time_block = {
            "slot": time_src.get("slot", base["time"]["slot"]),
            "time_str": time_src.get("time_str", base["time"]["time_str"]),
        }

        return {
            "locations": locations,
            "time": time_block,
        }

    # ---------------------------------------------------------
    # world_state の get / set
    # ---------------------------------------------------------
    def get_world_state(self) -> Dict[str, Any]:
        """llm_meta から world_state を取得（コピーして返す）。"""
        ws = self.llm_meta.get("world_state")
        if not isinstance(ws, dict):
            ws = self._default_world_state()
            self.llm_meta["world_state"] = ws
            self.state["llm_meta"] = self.llm_meta
        return {
            "locations": dict(ws.get("locations", {})),
            "time": dict(ws.get("time", {})),
        }

    def _save_world_state(self, ws: Dict[str, Any]) -> None:
        """world_state を llm_meta に書き戻す。"""
        self.llm_meta["world_state"] = self._merge_default_world_state(ws)
        self.state["llm_meta"] = self.llm_meta

    # ---------------------------------------------------------
    # プレイヤー / フローリアの移動
    # ---------------------------------------------------------
    def move_player(
        self,
        location: str,
        *,
        time_slot: Optional[str] = None,
        time_str: Optional[str] = None,
    ) -> None:
        """
        プレイヤーを指定場所に移動させる。
        time_slot / time_str が指定されていれば、世界共通の時間も更新。
        """
        ws = self.get_world_state()
        locs = ws.setdefault("locations", {})
        time_block = ws.setdefault("time", {})

        locs["player"] = str(location)

        if time_slot is not None:
            time_block["slot"] = str(time_slot)
        if time_str is not None:
            time_block["time_str"] = str(time_str)

        self._save_world_state(ws)

    def move_floria(self, location: str) -> None:
        """
        フローリアだけを別の場所へ移動させる。
        （時間帯・時刻は world 共通のまま）
        """
        ws = self.get_world_state()
        locs = ws.setdefault("locations", {})
        locs["floria"] = str(location)
        self._save_world_state(ws)

    # ---------------------------------------------------------
    # SceneManager から感情ボーナスを取得
    # ---------------------------------------------------------
    def get_emotion_bonus(self) -> Dict[str, float]:
        """
        現在の world_state（プレイヤーの場所＋時間）に対応する
        SceneManager 由来の感情ボーナスを返す。
        """
        ws = self.get_world_state()
        locs = ws.get("locations", {})
        t = ws.get("time", {})

        location = locs.get("player", "プレイヤーの部屋")
        slot_name = t.get("slot")
        time_str = t.get("time_str")

        return self.manager.get_for(
            location=location,
            time_str=time_str,
            slot_name=slot_name,
        )

    # ---------------------------------------------------------
    # LLM 向け：シーン説明 system プロンプト
    # ---------------------------------------------------------
    def build_scene_prompt_for_actor(self, actor_name: str = "フローリア") -> str:
        """
        現在の world_state をもとに、Actor / Composer 用の
        「シーン説明 system プロンプト」を組み立てる。
        （日本語で簡潔に）
        """
        ws = self.get_world_state()
        locs = ws.get("locations", {})
        t = ws.get("time", {})

        player_loc = locs.get("player", "プレイヤーの部屋")
        floria_loc = locs.get("floria", "プレイヤーの部屋")
        slot = t.get("slot", "morning")
        time_str = t.get("time_str", "07:30")

        slot_spec = self.manager.time_slots.get(slot, {})
        slot_label = f"{slot}（{slot_spec.get('start', '--:--')}〜{slot_spec.get('end', '--:--')}）"

        if player_loc == floria_loc:
            relation = "プレイヤーと{actor}は同じ場所にいて、直接会話できる状態です。".format(
                actor=actor_name
            )
        else:
            relation = (
                "プレイヤーと{actor}は別々の場所にいます。"
                "直接会話させる場合は、通信越し・魔法通信など、"
                "それらしい理由を補ってください。"
            ).format(actor=actor_name)

        prompt = f"""
あなたは、以下の「現在のシーン状況」を必ず踏まえて会話と描写を行うアシスタントです。

[現在のシーン]
- 時間帯: {slot_label}
- 時刻: {time_str}
- プレイヤーの場所: {player_loc}
- {actor_name}の場所: {floria_loc}
- 二人の位置関係: {relation}

上記と矛盾しないように、光景描写（天気・明るさ・人の多さなど）やセリフの内容を整合させてください。
"""
        return prompt.strip()
