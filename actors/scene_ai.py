# actors/scene_ai.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import os

import streamlit as st

from actors.scene.scene_manager import SceneManager
from actors.utils.debug_world_state import WorldStateDebugger


# 環境変数でデバッグ ON/OFF
LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"
_DEBUGGER = WorldStateDebugger(name="SceneAI")


@dataclass
class SceneAI:
    """
    シーン情報（world_state）の唯一の正規窓口。

    - world_state は state["world_state"] にだけ保持
    - SceneManager は state["scene_manager"] に 1 個だけ保持
    - 他クラス（NarratorAI / AnswerTalker / SceneManager UI など）は
      必ずこのクラス経由で world_state を取得する
    """

    state: Mapping[str, Any]

    def __init__(self, state: Optional[Mapping[str, Any]] = None) -> None:
        env_debug = os.getenv("LYRA_DEBUG", "")

        if state is not None:
            self.state = state
        elif env_debug == "1":
            self.state = st.session_state
        else:
            self.state = st.session_state

        # SceneManager をセッション内で 1 個だけ確保
        mgr_key = "scene_manager"
        mgr = self.state.get(mgr_key)
        if not isinstance(mgr, SceneManager):
            mgr = SceneManager(
                path="actors/scene/scene_bonus/scene_emotion_map.json"
            )
            mgr.load()
            # state は "Mapping" 型だが、実体は SessionState / dict を想定
            self.state[mgr_key] = mgr  # type: ignore[index]
        self.manager: SceneManager = mgr

        # world_state 初期化
        self._ensure_world_state_initialized()

    # ==========================================================
    # world_state の正規管理
    # ==========================================================
    def _ensure_world_state_initialized(self) -> None:
        """
        state["world_state"] がなければ、デフォルト値で初期化する。
        すでにあれば、不足フィールドだけを補完する。
        """
        ws = self.state.get("world_state")
        if not isinstance(ws, dict):
            ws = {}

        # locations
        loc = ws.get("locations") or {}
        if not isinstance(loc, dict):
            loc = {}

        # --- player name aware location ---
        player_name = ""
        if isinstance(self.state, dict):
            pn = self.state.get("player_name")
            if isinstance(pn, str) and pn.strip():
                player_name = pn.strip()

        default_player_loc = (
            f"{player_name}の部屋" if player_name else "プレイヤーの部屋"
        )

        player_loc = loc.get("player") or default_player_loc
        floria_loc = loc.get("floria") or player_loc

        loc["player"] = player_loc
        loc["floria"] = floria_loc

        # time
        t = ws.get("time") or {}
        if not isinstance(t, dict):
            t = {}
        slot = t.get("slot") or "morning"
        time_str = t.get("time_str") or "07:30"
        t["slot"] = slot
        t["time_str"] = time_str

        # ★ others_present（DokiPowerController からの manual override を反映）
        others_present: Any = ws.get("others_present")

        ws_manual = st.session_state.get("world_state_manual_controls")
        if isinstance(ws_manual, dict) and isinstance(
            ws_manual.get("others_present"), bool
        ):
            others_present = ws_manual["others_present"]

        if not isinstance(others_present, bool):
            others_present = False

        # weather
        weather = ws.get("weather") or "clear"

        # party（プレイヤー視点のパーティ状態）
        party = ws.get("party") or {}
        if not isinstance(party, dict):
            party = {}
        party_mode = self._calc_party_mode(player_loc, floria_loc)
        party["mode"] = party_mode

        # ★ dict の順序を意識してセット（time → others_present → weather）
        ws["locations"] = loc
        ws["time"] = t
        ws["others_present"] = others_present
        ws["weather"] = weather
        ws["party"] = party

        self.state["world_state"] = ws  # type: ignore[index]

    @staticmethod
    def _calc_party_mode(player_loc: Optional[str], floria_loc: Optional[str]) -> str:
        """
        プレイヤーから見たパーティ状態を文字列で返す。
        - "both"  : プレイヤーとフローリアが同じ場所
        - "alone" : プレイヤーのみ（またはフローリアが別の場所）
        """
        if not floria_loc:
            return "alone"
        if player_loc and player_loc == floria_loc:
            return "both"
        return "alone"

    def _sync_party_mode(self, ws: Dict[str, Any]) -> None:
        locs = ws.get("locations") or {}
        if not isinstance(locs, dict):
            locs = {}
        player_loc = locs.get("player")
        floria_loc = locs.get("floria")

        party = ws.get("party") or {}
        if not isinstance(party, dict):
            party = {}
        party["mode"] = self._calc_party_mode(player_loc, floria_loc)

        ws["locations"] = locs
        ws["party"] = party

    # ==========================================================
    # 公開 API：world_state 取得・更新
    # ==========================================================
    def get_world_state(self) -> Dict[str, Any]:
        """
        現在の world_state を dict で返す。
        呼び出し側から勝手に書き換えられないようコピーを返す。
        """
        self._ensure_world_state_initialized()
        ws = self.state.get("world_state") or {}
        if not isinstance(ws, dict):
            ws = {}
        # 浅いコピーで十分（ネスト内部は SceneAI 側でしか書き換えない前提）
        return dict(ws)

    def set_world_state(self, new_ws: Dict[str, Any]) -> None:
        """
        外部から丸ごと world_state をセットするためのフック。
        基本的には SceneAI 内部でのみ使用する想定。
        """
        if not isinstance(new_ws, dict):
            return
        # パーティ状態を再計算
        locs = new_ws.get("locations") or {}
        if not isinstance(locs, dict):
            locs = {}
        new_ws["locations"] = locs

        self._sync_party_mode(new_ws)
        # 最後に state へ格納
        self.state["world_state"] = new_ws  # type: ignore[index]

    # ----------------------------------------------------------
    # プレイヤー / フローリアの移動
    # ----------------------------------------------------------
    def move_player(
        self,
        location: str,
        *,
        time_slot: Optional[str] = None,
        time_str: Optional[str] = None,
    ) -> None:
        """
        プレイヤーのみを指定の場所＆時間に移動する。
        フローリアの位置は変更しない。
        """
        self._ensure_world_state_initialized()
        ws = self.state.get("world_state") or {}
        if not isinstance(ws, dict):
            ws = {}

        locs = ws.get("locations") or {}
        if not isinstance(locs, dict):
            locs = {}
        locs["player"] = location

        t = ws.get("time") or {}
        if not isinstance(t, dict):
            t = {}

        if time_slot is not None:
            t["slot"] = time_slot
        if time_str is not None:
            t["time_str"] = time_str

        ws["locations"] = locs
        ws["time"] = t

        # パーティ状態を更新
        self._sync_party_mode(ws)

        # 保存
        self.state["world_state"] = ws  # type: ignore[index]

    def move_floria(
        self,
        location: str,
        *,
        keep_time: bool = True,
    ) -> None:
        """
        フローリアだけを別の場所へ移動する。

        keep_time=True の場合、時間情報は変更しない。
        （今のところ時間の主導権はプレイヤー側が握る想定）
        """
        self._ensure_world_state_initialized()
        ws = self.state.get("world_state") or {}
        if not isinstance(ws, dict):
            ws = {}

        locs = ws.get("locations") or {}
        if not isinstance(locs, dict):
            locs = {}
        locs["floria"] = location
        ws["locations"] = locs

        # 時刻は原則そのまま
        if not keep_time:
            t = ws.get("time") or {}
            if not isinstance(t, dict):
                t = {}
            t.setdefault("slot", "morning")
            t.setdefault("time_str", "07:30")
            ws["time"] = t

        # パーティ状態を更新
        self._sync_party_mode(ws)

        self.state["world_state"] = ws  # type: ignore[index]

    # ==========================================================
    # SceneManager 連携：感情ボーナス
    # ==========================================================
    def get_scene_emotion(
        self,
        world_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        world_state をもとに SceneManager から感情補正ベクトルを取得。

        - 基本は「プレイヤーのいる場所」を基準に補正値を取る。
        """
        if world_state is None:
            world_state = self.get_world_state()

        locs = world_state.get("locations") or {}
        if not isinstance(locs, dict):
            locs = {}
        location = (
            locs.get("player")
            or locs.get("floria")
            or "プレイヤーの部屋"
        )

        t = world_state.get("time") or {}
        if not isinstance(t, dict):
            t = {}
        slot_name: Optional[str] = t.get("slot")
        time_str: Optional[str] = t.get("time_str")

        return self.manager.get_for(
            location=location,
            time_str=time_str,
            slot_name=slot_name,
        )

    def get_emotion_bonus(self) -> Dict[str, float]:
        """
        MixerAI から呼ばれる想定の薄いラッパ。
        現在の world_state に対する感情補正ベクトルを返す。
        """
        return self.get_scene_emotion()

    # ==========================================================
    # MixerAI / AnswerTalker 用 payload
    # ==========================================================
    def build_emotion_override_payload(self) -> Dict[str, Any]:
        """
        MixerAI や AnswerTalker へまとめて渡しやすい形。
        """
        ws = self.get_world_state()
        emo = self.get_scene_emotion(ws)

        # デバッグ出力
        _DEBUGGER.log(
            caller="SceneAI.build_emotion_override_payload",
            world_state=ws,
            scene_emotion=emo,
        )

        return {
            "world_state": ws,
            "scene_emotion": emo,
        }
