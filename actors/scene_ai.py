from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import os
import streamlit as st

from actors.scene.scene_manager import SceneManager
from actors.utils.debug_world_state import WorldStateDebugger
from actors.init_ai import InitAI
from actors.scene.world_context import WorldContext


# 環境変数でデバッグ ON/OFF
LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"
_DEBUGGER = WorldStateDebugger(name="SceneAI")


@dataclass
class SceneAI:
    """
    シーン情報（world_state）の唯一の正規窓口。

    - world_state は state["world_state"] にだけ保持
    - SceneManager は state["scene_manager"] に 1 個だけ保持
    - 初期化（world_state / manual_controls）は InitAI に一本化
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
            mgr = SceneManager(path="actors/scene/scene_bonus/scene_emotion_map.json")
            mgr.load()
            self.state[mgr_key] = mgr  # type: ignore[index]
        self.manager: SceneManager = mgr

        # world_state / manual_controls 初期化（InitAI に委譲）
        self._ensure_initialized()

    # ==========================================================
    # InitAI による初期化（一本化）
    # ==========================================================
    def _ensure_initialized(self) -> None:
        """
        world_state が未初期化なら InitAI.apply(WorldContext) で初期化する。
        すでにあれば、不足キーだけを補完する（壊さない）。
        """
        ws = self.state.get("world_state")

        # --- 初回（未初期化） ---
        if not isinstance(ws, dict) or not ws:
            # プレイヤー名の既定取得（※ここは将来 PlayerProfile へ寄せてもOK）
            player_name = (
                (st.session_state.get("player_name") or "")
                if isinstance(st.session_state, dict) or hasattr(st.session_state, "get")
                else ""
            )
            player_name = str(player_name).strip() or "アツシ"

            # 相手名は当面の既定（将来 Persona から注入でOK）
            partner_name = str(
                st.session_state.get("partner_name", "フローリア")  # type: ignore[arg-type]
            ).strip() or "フローリア"

            home = f"{player_name}の部屋"
            ctx = WorldContext(
                player_name=player_name,
                partner_name=partner_name,
                player_location=home,
                partner_location=home,
                time_slot="morning",
                time_str="07:30",
                others_present=False,
                weather="clear",
            )
            InitAI.apply(ctx, state=self.state)
            return

        # --- 既存 ws がある場合：不足だけ補完（既存値は尊重） ---
        ws.setdefault("player_name", st.session_state.get("player_name", "アツシ"))  # type: ignore[arg-type]

        loc = ws.get("locations")
        if not isinstance(loc, dict):
            loc = {}
        player_name = str(ws.get("player_name") or "アツシ")
        home = f"{player_name}の部屋"
        loc.setdefault("player", home)
        loc.setdefault("floria", home)
        ws["locations"] = loc

        t = ws.get("time")
        if not isinstance(t, dict):
            t = {}
        t.setdefault("slot", "morning")
        t.setdefault("time_str", "07:30")
        ws["time"] = t

        # others_present（manual override があれば尊重）
        others_present = ws.get("others_present")
        ws_manual = st.session_state.get("world_state_manual_controls")
        if isinstance(ws_manual, dict) and isinstance(ws_manual.get("others_present"), bool):
            others_present = ws_manual["others_present"]
        if not isinstance(others_present, bool):
            others_present = False
        ws["others_present"] = others_present

        ws.setdefault("weather", "clear")

        party = ws.get("party")
        if not isinstance(party, dict):
            party = {}
        party["mode"] = self._calc_party_mode(loc.get("player"), loc.get("floria"))
        ws["party"] = party

        # state へ保存
        self.state["world_state"] = ws  # type: ignore[index]

        # manual_controls の最低保証（InitAI と同等のキーを揃える）
        if "world_state_manual_controls" not in st.session_state or not isinstance(
            st.session_state.get("world_state_manual_controls"), dict
        ):
            st.session_state["world_state_manual_controls"] = {
                "others_present": others_present,
                "interaction_mode_hint": "auto",
            }
        else:
            st.session_state["world_state_manual_controls"].setdefault("others_present", others_present)
            st.session_state["world_state_manual_controls"].setdefault("interaction_mode_hint", "auto")

        if "emotion_manual_controls" not in st.session_state or not isinstance(
            st.session_state.get("emotion_manual_controls"), dict
        ):
            st.session_state["emotion_manual_controls"] = {
                "environment": "with_others" if others_present else "alone",
                "others_present": others_present,
                "interaction_mode_hint": "auto",
            }
        else:
            st.session_state["emotion_manual_controls"].setdefault(
                "environment", "with_others" if others_present else "alone"
            )
            st.session_state["emotion_manual_controls"].setdefault("others_present", others_present)
            st.session_state["emotion_manual_controls"].setdefault("interaction_mode_hint", "auto")

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
        self._ensure_initialized()
        ws = self.state.get("world_state") or {}
        if not isinstance(ws, dict):
            ws = {}
        return dict(ws)

    def set_world_state(self, new_ws: Dict[str, Any]) -> None:
        """
        外部から丸ごと world_state をセットするためのフック。
        """
        if not isinstance(new_ws, dict):
            return

        locs = new_ws.get("locations") or {}
        if not isinstance(locs, dict):
            locs = {}
        new_ws["locations"] = locs

        self._sync_party_mode(new_ws)
        self.state["world_state"] = new_ws  # type: ignore[index]

        # 反映後も最低限の初期保証
        self._ensure_initialized()

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
        self._ensure_initialized()
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
        self._sync_party_mode(ws)
        self.state["world_state"] = ws  # type: ignore[index]

    def move_floria(
        self,
        location: str,
        *,
        keep_time: bool = True,
    ) -> None:
        self._ensure_initialized()
        ws = self.state.get("world_state") or {}
        if not isinstance(ws, dict):
            ws = {}

        locs = ws.get("locations") or {}
        if not isinstance(locs, dict):
            locs = {}
        locs["floria"] = location
        ws["locations"] = locs

        if not keep_time:
            t = ws.get("time") or {}
            if not isinstance(t, dict):
                t = {}
            t.setdefault("slot", "morning")
            t.setdefault("time_str", "07:30")
            ws["time"] = t

        self._sync_party_mode(ws)
        self.state["world_state"] = ws  # type: ignore[index]

    # ==========================================================
    # SceneManager 連携：感情ボーナス
    # ==========================================================
    def get_scene_emotion(
        self,
        world_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        if world_state is None:
            world_state = self.get_world_state()

        locs = world_state.get("locations") or {}
        if not isinstance(locs, dict):
            locs = {}

        location = locs.get("player") or locs.get("floria") or "プレイヤーの部屋"

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
        return self.get_scene_emotion()

    # ==========================================================
    # MixerAI / AnswerTalker 用 payload
    # ==========================================================
    def build_emotion_override_payload(self) -> Dict[str, Any]:
        ws = self.get_world_state()
        emo = self.get_scene_emotion(ws)

        _DEBUGGER.log(
            caller="SceneAI.build_emotion_override_payload",
            world_state=ws,
            scene_emotion=emo,
        )

        return {
            "world_state": ws,
            "scene_emotion": emo,
        }
