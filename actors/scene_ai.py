# actors/scene_ai.py
from __future__ import annotations

from typing import Any, Dict, Optional, Mapping
import os

import streamlit as st

from actors.scene.scene_manager import SceneManager


class SceneAI:
    """
    シーン情報（場所・時間帯）と「誰がどこにいるか」を一元管理するクラス。

    - world_state の唯一のソース
    - SceneManager からシーン補正ベクトルを取得
    - MixerAI 向けの emotion_bonus
    - AnswerTalker / Composer 向けの world_state / scene_emotion ペイロード
    """

    DEFAULT_PLAYER_LOC = "プレイヤーの部屋"
    DEFAULT_FLORIA_LOC = "プレイヤーの部屋"
    DEFAULT_SLOT = "morning"
    DEFAULT_TIME_STR = "07:30"

    def __init__(self, state: Optional[Mapping[str, Any]] = None) -> None:
        env_debug = os.getenv("LYRA_DEBUG", "")

        if state is not None:
            self.state = state  # 明示 state 優先
        elif env_debug == "1":
            self.state = st.session_state
        else:
            self.state = st.session_state

        # SceneManager をセッション内で 1 個だけ確保
        key_mgr = "scene_manager"
        if key_mgr not in self.state:
            mgr = SceneManager(
                path="actors/scene/scene_bonus/scene_emotion_map.json"
            )
            mgr.load()
            self.state[key_mgr] = mgr

        self.manager: SceneManager = self.state[key_mgr]

        # llm_meta 参照（あれば使う）
        llm_meta = self.state.get("llm_meta")
        if not isinstance(llm_meta, dict):
            llm_meta = {}
        self.llm_meta: Dict[str, Any] = llm_meta

    # =========================================================
    # world_state のソース
    # =========================================================
    def _ensure_world_state(self) -> Dict[str, Any]:
        """
        state["world_state"] が無ければ初期値を入れて返す。
        以後はここが唯一のソースになる。
        """
        ws = self.state.get("world_state")
        if not isinstance(ws, dict):
            ws = {}
        locs = ws.setdefault("locations", {})
        time = ws.setdefault("time", {})

        locs.setdefault("player", self.DEFAULT_PLAYER_LOC)
        # 初期状態ではフローリアも同じ場所にいる想定
        locs.setdefault("floria", self.DEFAULT_FLORIA_LOC)

        time.setdefault("slot", self.DEFAULT_SLOT)
        time.setdefault("time_str", self.DEFAULT_TIME_STR)

        # party.mode を更新
        self._update_party_mode(ws)

        self.state["world_state"] = ws
        return ws

    def _update_party_mode(self, ws: Dict[str, Any]) -> None:
        locs = ws.setdefault("locations", {})
        player_loc = locs.get("player")
        floria_loc = locs.get("floria")

        if player_loc and floria_loc and player_loc == floria_loc:
            mode = "with_floria"
        else:
            mode = "alone"

        ws.setdefault("party", {})
        ws["party"]["mode"] = mode

    # ---------------------------------------------------------
    # 公開 API: world_state 取得
    # ---------------------------------------------------------
    def get_world_state(self) -> Dict[str, Any]:
        """
        現在の world_state を返す。
        - locations: {player, floria}
        - time: {slot, time_str}
        - party: {mode: "alone" / "with_floria"}
        """
        ws = self._ensure_world_state()
        # 念のため毎回 party.mode を更新
        self._update_party_mode(ws)

        # llm_meta にも同期（あれば）
        self.llm_meta["world_state"] = ws
        self.state["llm_meta"] = self.llm_meta

        return ws

    # =========================================================
    # 移動 API
    # =========================================================
    def move_player(
        self,
        location: str,
        *,
        time_slot: Optional[str] = None,
        time_str: Optional[str] = None,
    ) -> None:
        """
        プレイヤーだけを指定の場所へ移動させる。
        フローリアは動かさない。
        """
        ws = self._ensure_world_state()
        locs = ws.setdefault("locations", {})
        time = ws.setdefault("time", {})

        locs["player"] = location
        if time_slot is not None:
            time["slot"] = time_slot
        if time_str is not None:
            time["time_str"] = time_str

        self._update_party_mode(ws)
        self.state["world_state"] = ws

        self.llm_meta["world_state"] = ws
        self.state["llm_meta"] = self.llm_meta

    def move_floria(
        self,
        location: str,
    ) -> None:
        """
        フローリアだけを指定の場所へ移動させる。
        時刻情報は変更しない。
        """
        ws = self._ensure_world_state()
        locs = ws.setdefault("locations", {})

        locs["floria"] = location

        self._update_party_mode(ws)
        self.state["world_state"] = ws

        self.llm_meta["world_state"] = ws
        self.state["llm_meta"] = self.llm_meta

    # =========================================================
    # シーン感情ボーナス
    # =========================================================
    def _pick_slot_and_time(self, ws: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        time = ws.get("time", {})
        slot = time.get("slot")
        time_str = time.get("time_str")
        return slot, time_str

    def get_scene_emotion_for_location(
        self,
        location: str,
        *,
        slot_name: Optional[str],
        time_str: Optional[str],
    ) -> Dict[str, float]:
        """
        SceneManager から「指定ロケーションのシーン補正ベクトル」を取得。
        """
        return self.manager.get_for(
            location=location,
            time_str=time_str,
            slot_name=slot_name,
        )

    def get_emotion_bonus(self) -> Dict[str, float]:
        """
        MixerAI 用の「シーン補正ベクトル」。

        方針：
        - フローリアがプレイヤーと同じ場所にいるとき：
            → その場所＆時刻の補正ベクトル
        - 離れているとき：
            → 0 ベクトル（会話してない前提）
        """
        ws = self.get_world_state()
        locs = ws.get("locations", {})
        player_loc = locs.get("player", self.DEFAULT_PLAYER_LOC)
        floria_loc = locs.get("floria", self.DEFAULT_FLORIA_LOC)

        slot, time_str = self._pick_slot_and_time(ws)

        if floria_loc and floria_loc == player_loc:
            return self.get_scene_emotion_for_location(
                location=player_loc,
                slot_name=slot,
                time_str=time_str,
            )

        # 離れているときは 0 ベクトル
        return {dim: 0.0 for dim in self.manager.dimensions}

    # =========================================================
    # AnswerTalker / Composer 向けペイロード
    # =========================================================
    def build_emotion_override_payload(self) -> Dict[str, Any]:
        """
        AnswerTalker から呼ばれることを想定。
        - world_state
        - scene_emotion（会話に効くシーン補正）
        をまとめて返し、llm_meta にも突っ込む。
        """
        ws = self.get_world_state()
        scene_emo = self.get_emotion_bonus()

        self.llm_meta["world_state"] = ws
        self.llm_meta["scene_emotion"] = scene_emo
        self.state["llm_meta"] = self.llm_meta

        return {
            "world_state": ws,
            "scene_emotion": scene_emo,
        }
