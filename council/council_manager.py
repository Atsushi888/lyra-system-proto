from __future__ import annotations

from typing import List, Dict, Any, Optional

import os
import streamlit as st

from actors.actor import Actor
from actors.persona.persona_classes.persona_riseria_ja import Persona as RiseriaPersona
from actors.narrator_ai import NarratorAI
from actors.narrator.narrator_manager import NarratorManager
from actors.scene_ai import SceneAI


LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"


def _dwrite(*args: Any) -> None:
    if LYRA_DEBUG:
        st.write(*args)


# ==========================================================
# CouncilManager を取得するヘルパ
# ==========================================================
def get_or_create_riseria_council_manager(player_name: str = "アツシ") -> "CouncilManager":
    key_riseria = "council_manager_riseria"
    key_generic = "council_manager"

    if key_riseria not in st.session_state:
        _dwrite(f"[DEBUG:Council] create CouncilManager for Riseria (player_name={player_name})")

        riseria_persona = RiseriaPersona(player_name=player_name)
        riseria_actor = Actor(
            name=riseria_persona.display_name,
            persona=riseria_persona,
        )

        manager = CouncilManager(
            partner=riseria_actor,
            partner_role="riseria",
            session_key="council_log_riseria",
        )

        st.session_state[key_riseria] = manager
        st.session_state[key_generic] = manager
    else:
        if key_generic not in st.session_state:
            st.session_state[key_generic] = st.session_state[key_riseria]

    return st.session_state[key_riseria]


# ==========================================================
# CouncilManager 本体
# ==========================================================
class CouncilManager:
    """
    会談システムのロジック＋UI。
    - player 表示名は session_state["player_name"] を優先
    - sending フラグが True のまま残って UI が死なないよう try/finally で必ず戻す
    """

    def __init__(
        self,
        partner: Optional[Actor] = None,
        partner_role: Optional[str] = None,
        session_key: str = "council_log",
    ) -> None:
        self.session_key = session_key

        # ログ
        raw_log = st.session_state.get(self.session_key, [])
        self.conversation_log: List[Dict[str, str]] = list(raw_log) if isinstance(raw_log, list) else []

        # partner
        if partner is None:
            # 互換（通常ここは使わない想定）
            from personas.persona_floria_ja import Persona as FloriaPersona  # type: ignore
            partner = Actor("フローリア", FloriaPersona())
            partner_role = "floria"
        else:
            if partner_role is None:
                partner_role = "partner"

        self.partner_role: str = str(partner_role)
        self.partner: Actor = partner

        self.actors: Dict[str, Actor] = {self.partner_role: self.partner}

        # 状態
        self.state: Dict[str, Any] = {
            "mode": "ongoing",
            "participants": ["player", self.partner_role],
            "last_speaker": self.conversation_log[-1]["role"] if self.conversation_log else None,
            "round0_done": bool(self.conversation_log),
            "special_available": False,
            "special_id": None,
        }

        # world_state 初期化（必ず）
        SceneAI(state=st.session_state)

        # narrator
        if "narrator_manager" not in st.session_state:
            st.session_state["narrator_manager"] = NarratorManager(state=st.session_state)
        self.narrator_manager: NarratorManager = st.session_state["narrator_manager"]

        self.narrator = NarratorAI(
            manager=self.narrator_manager,
            partner_role=self.partner_role,
            partner_name=getattr(self.partner, "name", self.partner_role),
        )

        self._ensure_round0_initialized()

    # ------------------------------------------------------
    # world_state helpers
    # ------------------------------------------------------
    def _get_world_snapshot(self) -> Dict[str, Any]:
        llm_meta = st.session_state.get("llm_meta", {})
        if isinstance(llm_meta, dict):
            world = llm_meta.get("world_state") or llm_meta.get("world") or {}
            if isinstance(world, dict) and world:
                return world

        scene_ai = SceneAI(state=st.session_state)
        return scene_ai.get_world_state()

    def _build_narrator_world_state(self) -> Dict[str, Any]:
        world = self._get_world_snapshot()
        locs = world.get("locations", {}) if isinstance(world.get("locations", {}), dict) else {}
        t = world.get("time", {}) if isinstance(world.get("time", {}), dict) else {}

        location_name = locs.get("player") or "通学路"
        time_of_day = t.get("slot", "morning")
        weather = world.get("weather", "clear")

        return {
            "location_name": location_name,
            "time_of_day": time_of_day,
            "weather": weather,
        }

    # ------------------------------------------------------
    # log
    # ------------------------------------------------------
    def _save_log_to_session(self) -> None:
        st.session_state[self.session_key] = list(self.conversation_log)

    def _append_log(self, role: str, content: str) -> None:
        safe = (content or "").replace("\n", "<br>")
        self.conversation_log.append({"role": role, "content": safe})
        self.state["last_speaker"] = role
        self._save_log_to_session()

    def _ensure_round0_initialized(self) -> None:
        if self.state.get("round0_done", False):
            return

        world_state = self._build_narrator_world_state()
        player_profile: Dict[str, Any] = {}
        partner_state = {"mood": "slightly_nervous"}

        text = ""
        try:
            line = self.narrator.generate_round0_opening(
                world_state=world_state,
                player_profile=player_profile,
                floria_state=partner_state,
            )
            text = (getattr(line, "text", None) or "").strip()
        except Exception as e:
            if LYRA_DEBUG:
                st.exception(e)

        if not text:
            text = f"{getattr(self.partner, 'name', 'その子')}は、どこかそわそわした様子であなたの前に立っている。"

        self._append_log("narrator", text)
        self.state["round0_done"] = True

    # ------------------------------------------------------
    # public
    # ------------------------------------------------------
    def reset(self) -> None:
        self.conversation_log.clear()
        self.state["mode"] = "ongoing"
        self.state["last_speaker"] = None
        self.state["round0_done"] = False
        self.state["special_available"] = False
        self.state["special_id"] = None

        st.session_state.pop("council_rescue_buffer", None)
        st.session_state.pop("council_pending_action", None)
        st.session_state.pop("council_sending", None)
        st.session_state.pop("council_rescue_running", None)

        self._save_log_to_session()
        self._ensure_round0_initialized()

    def get_log(self) -> List[Dict[str, str]]:
        return list(self.conversation_log)

    def get_status(self) -> Dict[str, Any]:
        round_ = len(self.conversation_log) + 1
        world = self._get_world_snapshot()
        locs = world.get("locations", {}) if isinstance(world.get("locations", {}), dict) else {}
        t = world.get("time", {}) if isinstance(world.get("time", {}), dict) else {}

        partner_loc = locs.get(self.partner_role) or locs.get("floria")

        return {
            "round": round_,
            "speaker": "player",
            "mode": self.state.get("mode", "ongoing"),
            "participants": self.state.get("participants", ["player", self.partner_role]),
            "last_speaker": self.state.get("last_speaker"),
            "special_available": self.state.get("special_available", False),
            "world": {
                "player_location": locs.get("player"),
                "partner_location": partner_loc,
                "time_slot": t.get("slot"),
                "time_str": t.get("time_str"),
            },
        }

    def proceed(self, user_text: str) -> str:
        self._append_log("player", user_text)

        reply = ""
        actor = self.actors.get(self.partner_role)
        if actor is not None:
            reply = actor.speak(self.conversation_log)  # Actor 側の実装に合わせる
            self._append_log(self.partner_role, reply)

        return reply

    # ------------------------------------------------------
    # rescue
    # ------------------------------------------------------
    def build_rescue_text(self, kind: str) -> str:
        world_state = self._build_narrator_world_state()
        partner_state = {"mood": "slightly_nervous"}

        if kind == "wait":
            choice = self.narrator.make_wait_choice(world_state, partner_state)
        elif kind == "look_person":
            choice = self.narrator.make_look_person_choice(
                actor_name=getattr(self.partner, "name", "相手"),
                world_state=world_state,
                floria_state=partner_state,
            )
        elif kind == "scan_area":
            choice = self.narrator.make_scan_area_choice(
                location_name=world_state["location_name"],
                world_state=world_state,
                floria_state=partner_state,
            )
        elif kind == "special":
            special_id = self.state.get("special_id") or "unknown_special"
            _, choice = self.narrator.make_special_title_and_choice(
                special_id,
                world_state=world_state,
                floria_state=partner_state,
            )
        else:
            return ""

        return choice.speak_text or ""

    # ------------------------------------------------------
    # UI
    # ------------------------------------------------------
    def render(self) -> None:
        # --- flags ---
        if "council_sending" not in st.session_state:
            st.session_state["council_sending"] = False
        if "council_pending_action" not in st.session_state:
            st.session_state["council_pending_action"] = None
        if "council_rescue_running" not in st.session_state:
            st.session_state["council_rescue_running"] = False

        sending: bool = bool(st.session_state.get("council_sending", False))

        # 表示名
        player_name = st.session_state.get("player_name", "アツシ")
        partner_name = getattr(self.partner, "name", self.partner_role)

        log = self.get_log()
        status = self.get_status()
        world_info = status.get("world", {}) or {}

        st.markdown("## 🗣️ 会談システム（Council Prototype）")
        st.caption("※ Actor ベースで AI と会話する会談システム（β）です。")

        col_left, col_right = st.columns([3, 1])
        with col_right:
            if st.button("🔁 リセット", key="council_reset"):
                self.reset()
                st.success("会談をリセットしました。")
                st.rerun()

        # ---- 会談ログ ----
        st.markdown("### 会談ログ")
        if not log:
            st.caption("（まだ会談は始まっていません。何か話しかけてみましょう）")
        else:
            for idx, entry in enumerate(log, start=1):
                role = entry.get("role", "")
                text = entry.get("content", "")

                if role == "player":
                    name = player_name
                elif role == "narrator":
                    name = "ナレーション"
                elif role == self.partner_role:
                    name = partner_name
                else:
                    # 互換：未知 role はそのまま出す
                    name = role or "？"

                st.markdown(f"**[{idx}] {name}**")
                st.markdown(text, unsafe_allow_html=True)
                st.markdown("---")

        # ---- サイドバー：会談ステータス ----
        with st.sidebar.expander("📊 会談ステータス", expanded=True):
            st.write(f"ラウンド: {status.get('round')}")
            st.write(f"話者: {player_name}")
            st.write(f"モード: {status.get('mode')}")
            st.write(f"相手: {partner_name}")
            st.write(f"スペシャル選択可: {status.get('special_available')}")
            st.markdown("---")
            st.write("**現在の世界情報**")
            st.write(f"プレイヤー位置: {world_info.get('player_location')}")
            st.write(f"相手位置: {world_info.get('partner_location')}")
            st.write(f"時間帯: {world_info.get('time_slot')}")
            st.write(f"時刻: {world_info.get('time_str')}")

            # “送信ロックが残ってる” 事故の復旧スイッチ（地味に効く）
            st.markdown("---")
            if sending:
                if st.button("🧯 送信ロック解除（復旧）", key="council_unlock"):
                    st.session_state["council_sending"] = False
                    st.session_state["council_rescue_running"] = False
                    st.success("解除しました。")
                    st.rerun()

        # ---- プレイヤー入力 ----
        st.markdown("### あなたの発言")

        round_no = int(status.get("round") or 1)
        input_key = f"council_user_input_r{round_no}"

        buffer = st.session_state.get("council_rescue_buffer")
        if isinstance(buffer, dict) and buffer.get("round") == round_no:
            st.session_state[input_key] = buffer.get("text", "")
            st.session_state["council_rescue_buffer"] = None

        user_text = st.text_area(
            "あなたの発言：",
            key=input_key,
            placeholder=f"ここに{partner_name}への発言を書いてください。",
            disabled=sending,
        )

        send_col, wait_col, look_col, scan_col, special_col = st.columns([1, 1, 1, 1, 1])

        with send_col:
            send_clicked = st.button("送信", key="council_send", disabled=sending)

        # 救済は「消えない」よう常に表示（sending 中は押せない）
        with wait_col:
            wait_clicked = st.button("何もしない", key="council_wait", disabled=sending)
        with look_col:
            look_clicked = st.button("相手の様子を伺う", key="council_look", disabled=sending)
        with scan_col:
            scan_clicked = st.button("周りの様子を見る", key="council_scan", disabled=sending)
        with special_col:
            special_clicked = st.button("スペシャル", key="council_special", disabled=sending)

        # --- actions ---
        if send_clicked:
            cleaned = (user_text or "").strip()
            if not cleaned:
                st.warning("発言を入力してください。")
            else:
                st.session_state["council_sending"] = True
                try:
                    with st.spinner(f"{partner_name}は少し考えています…"):
                        self.proceed(cleaned)
                finally:
                    # ★ 例外が起きても必ず解除（救済ボタン消失/操作不能の根本対策）
                    st.session_state["council_sending"] = False
                st.rerun()

        if wait_clicked:
            st.session_state["council_pending_action"] = "wait"
            st.rerun()
        if look_clicked:
            st.session_state["council_pending_action"] = "look_person"
            st.rerun()
        if scan_clicked:
            st.session_state["council_pending_action"] = "scan_area"
            st.rerun()
        if special_clicked:
            if not self.state.get("special_available", False):
                st.info("ここでスペシャルは選択できません。")
            else:
                st.session_state["council_pending_action"] = "special"
                st.rerun()

        # --- pending rescue confirm ---
        pending = st.session_state.get("council_pending_action")
        if pending:
            if pending == "wait":
                msg = "このターンは何も行動せず、様子を見ます。よろしいですか？"
            elif pending == "look_person":
                msg = "隣にいる相手の様子をうかがいます。よろしいですか？"
            elif pending == "scan_area":
                msg = "周囲の様子を見回します。よろしいですか？"
            elif pending == "special":
                msg = "スペシャルアクションを実行します。よろしいですか？"
            else:
                msg = "この行動を実行します。よろしいですか？"

            st.markdown("---")
            st.warning(msg)

            col_ok, col_cancel = st.columns(2)
            with col_ok:
                ok_clicked = st.button("実行する", key="council_rescue_ok")
            with col_cancel:
                cancel_clicked = st.button("キャンセル", key="council_rescue_cancel")

            if ok_clicked:
                st.session_state["council_rescue_running"] = True
                try:
                    with st.spinner("ナレーション案を考えています…"):
                        text = self.build_rescue_text(str(pending))
                    st.session_state["council_rescue_buffer"] = {"round": round_no, "text": text}
                finally:
                    st.session_state["council_rescue_running"] = False
                    st.session_state["council_pending_action"] = None
                st.rerun()

            if cancel_clicked:
                st.session_state["council_pending_action"] = None
                st.rerun()
