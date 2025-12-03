# council/council_manager.py

from __future__ import annotations
from typing import List, Dict, Any

import streamlit as st

from actors.actor import Actor
from personas.persona_floria_ja import Persona
from actors.narrator_ai import NarratorAI
from actors.narrator.narrator_manager import NarratorManager
from actors.scene_ai import SceneAI


def get_or_create_council_actor() -> Actor:
    """
    会談システム用の Actor を1つだけ生成・再利用する。
    """
    actor_key = "council_actor"

    if actor_key not in st.session_state:
        st.session_state[actor_key] = Actor(
            name="フローリア",
            persona=Persona(),
        )

    return st.session_state[actor_key]


class CouncilManager:
    """
    会談システムのロジック ＋ 画面描画（β）。
    """

    def __init__(self) -> None:
        self.conversation_log: List[Dict[str, str]] = []

        self.actors: Dict[str, Actor] = {
            "floria": Actor("フローリア", Persona())
        }

        self.state: Dict[str, Any] = {
            "mode": "ongoing",
            "participants": ["player", "floria"],
            "last_speaker": None,
            "round0_done": False,
            "special_available": False,
            "special_id": None,
        }

        if "narrator_manager" not in st.session_state:
            st.session_state["narrator_manager"] = NarratorManager(state=st.session_state)
        self.narrator_manager: NarratorManager = st.session_state["narrator_manager"]

        self.narrator = NarratorAI(manager=self.narrator_manager)

        # Round0 を 1 回だけ差し込む
        self._ensure_round0_initialized()

    # ===== 内部ヘルパ =====
    def _append_log(self, role: str, content: str) -> None:
        safe = (content or "").replace("\n", "<br>")
        self.conversation_log.append({"role": role, "content": safe})
        self.state["last_speaker"] = role

    def _get_world_state_for_narrator(self) -> Dict[str, Any]:
        """
        SceneAI から world_state を取得し、
        NarratorAI 用の world_state に変換する。
        """
        scene_ai = SceneAI(state=st.session_state)
        ws = scene_ai.get_world_state()

        location = ws.get("location", "通学路")
        time_slot = ws.get("time_slot") or "night"
        time_str = ws.get("time_str") or ""

        return {
            "location_name": location,
            "time_of_day": time_slot,
            "time_str": time_str,
            "weather": "clear",
        }

    def _ensure_round0_initialized(self) -> None:
        """
        会談開始時に Round0 ナレーションを 1 回だけ差し込む。
        conversation_log が空のときのみ生成する。
        """
        if self.conversation_log:
            return

        world_state = self._get_world_state_for_narrator()
        player_profile: Dict[str, Any] = {}
        floria_state = {"mood": "slightly_nervous"}

        line = self.narrator.generate_round0_opening(
            world_state=world_state,
            player_profile=player_profile,
            floria_state=floria_state,
        )
        self._append_log("narrator", line.text)
        self.state["round0_done"] = True

    # ===== ロジック側 公開 API =====
    def reset(self) -> None:
        """会談を最初からやり直す。"""
        self.conversation_log.clear()
        self.state["mode"] = "ongoing"
        self.state["last_speaker"] = None
        self.state["round0_done"] = False
        self.state["special_available"] = False
        self.state["special_id"] = None

        st.session_state.pop("council_rescue_buffer", None)
        st.session_state.pop("council_pending_action", None)

        self._ensure_round0_initialized()

    def get_log(self) -> List[Dict[str, str]]:
        return list(self.conversation_log)

    def get_status(self) -> Dict[str, Any]:
        round_ = len(self.conversation_log) + 1

        return {
            "round": round_,
            "speaker": "player",
            "mode": self.state.get("mode", "ongoing"),
            "participants": self.state.get("participants", ["player", "floria"]),
            "last_speaker": self.state.get("last_speaker"),
            "special_available": self.state.get("special_available", False),
        }

    def proceed(self, user_text: str) -> str:
        self._append_log("player", user_text)

        reply = ""
        actor = self.actors.get("floria")
        if actor is not None:
            reply = actor.speak(self.conversation_log)
            self._append_log("floria", reply)

        return reply

    # ===== 救済アクション処理 =====
    def build_rescue_text(self, kind: str) -> str:
        """
        救済ボタンからの行動を処理し、
        プレイヤー用ナレーション（地の文）だけを返す。
        """
        world_state = self._get_world_state_for_narrator()
        floria_state = {"mood": "slightly_nervous"}

        if kind == "wait":
            choice = self.narrator.make_wait_choice(world_state, floria_state)

        elif kind == "look_person":
            choice = self.narrator.make_look_person_choice(
                actor_name="フローリア",
                world_state=world_state,
                floria_state=floria_state,
            )

        elif kind == "scan_area":
            choice = self.narrator.make_scan_area_choice(
                location_name=world_state["location_name"],
                world_state=world_state,
                floria_state=floria_state,
            )

        elif kind == "special":
            special_id = self.state.get("special_id") or "unknown_special"
            _, choice = self.narrator.make_special_title_and_choice(
                special_id,
                world_state=world_state,
                floria_state=floria_state,
            )
        else:
            return ""

        return choice.speak_text or ""

    # ===== 画面描画 =====
    def render(self) -> None:
        # world_state が変更されたら会話をリセットして Round0 から
        if st.session_state.get("world_state_changed"):
            self.reset()
            st.session_state["world_state_changed"] = False

        if "council_sending" not in st.session_state:
            st.session_state["council_sending"] = False
        if "council_pending_action" not in st.session_state:
            st.session_state["council_pending_action"] = None
        if "council_rescue_running" not in st.session_state:
            st.session_state["council_rescue_running"] = False

        sending: bool = st.session_state["council_sending"]

        log = self.get_log()
        status = self.get_status()

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
                    name = "プレイヤー"
                elif role == "floria":
                    name = "フローリア"
                elif role == "narrator":
                    name = "ナレーション"
                else:
                    name = role or "？"

                st.markdown(f"**[{idx}] {name}**")
                st.markdown(text, unsafe_allow_html=True)
                st.markdown("---")

        # ---- サイドバー：会談ステータス ＋ world_state ----
        with st.sidebar.expander("📊 会談ステータス", expanded=True):
            st.write(f"ラウンド: {status.get('round')}")
            st.write(f"話者: {status.get('speaker')}")
            st.write(f"モード: {status.get('mode')}")
            participants = status.get("participants") or []
            if participants:
                st.write("参加者: " + " / ".join(participants))
            last = status.get("last_speaker")
            if last:
                st.write(f"最後の話者: {last}")
            st.write(f"スペシャル選択可: {status.get('special_available')}")

            # world_state 表示
            st.markdown("---")
            scene_ai = SceneAI(state=st.session_state)
            ws = scene_ai.get_world_state()
            st.write("**現在の world_state**")
            st.write(f"- 場所: {ws.get('location', '不明')}")
            st.write(f"- 時間帯スロット: {ws.get('time_slot') or 'auto'}")
            st.write(f"- 時刻: {ws.get('time_str') or '（未設定）'}")

        # ---- プレイヤー入力 ----
        st.markdown("### プレイヤー入力")

        round_no = int(status.get("round") or 1)
        input_key = f"council_user_input_r{round_no}"

        buffer = st.session_state.get("council_rescue_buffer")
        if isinstance(buffer, dict):
            if buffer.get("round") == round_no:
                st.session_state[input_key] = buffer.get("text", "")
                st.session_state["council_rescue_buffer"] = None

        user_text = st.text_area(
            "あなたの発言：",
            key=input_key,
            placeholder="ここにフローリアへの発言を書いてください。",
        )

        send_col, wait_col, look_col, scan_col, special_col = st.columns([1, 1, 1, 1, 1])

        with send_col:
            send_clicked = st.button(
                "送信",
                key="council_send",
                disabled=sending,
            )

        with wait_col:
            wait_clicked = st.button(
                "何もしない",
                key="council_wait",
                disabled=sending,
            )
        with look_col:
            look_clicked = st.button(
                "相手の様子を伺う",
                key="council_look",
                disabled=sending,
            )
        with scan_col:
            scan_clicked = st.button(
                "周りの様子を見る",
                key="council_scan",
                disabled=sending,
            )
        with special_col:
            special_clicked = st.button(
                "スペシャル",
                key="council_special",
                disabled=sending,
            )

        # ---- 通常送信処理 ----
        if send_clicked:
            cleaned = (user_text or "").strip()
            if not cleaned:
                st.warning("発言を入力してください。")
            else:
                if st.session_state["council_sending"]:
                    st.info("いま処理中です。少し待ってから再度お試しください。")
                else:
                    st.session_state["council_sending"] = True

                    with st.spinner("フローリアは少し考えています…"):
                        self.proceed(cleaned)

                    st.session_state["council_sending"] = False
                    st.rerun()

        # ---- 救済ボタン → pending_action へ ----
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

        # ---- 救済アクションの確認ウインドウ ----
        pending = st.session_state.get("council_pending_action")
        if pending:
            if pending == "wait":
                msg = "このターンは何も行動せず、様子を見ます。よろしいですか？"
            elif pending == "look_person":
                msg = "隣にいる相手の様子をうかがいます。よろしいですか？"
            elif pending == "scan_area":
                msg = "周囲の様子を見回します。よろしいですか？"
            elif pending == "special":
                special_id = self.state.get("special_id") or "unknown_special"
                title, _ = self.narrator.make_special_title_and_choice(
                    special_id,
                    world_state=self._get_world_state_for_narrator(),
                    floria_state={"mood": "slightly_nervous"},
                )
                msg = f"スペシャルアクション「{title}」を実行します。よろしいですか？"
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
                if st.session_state["council_rescue_running"]:
                    st.info("救済アクションを処理中です。少し待ってください。")
                else:
                    st.session_state["council_rescue_running"] = True
                    with st.spinner("ナレーション案を考えています…"):
                        text = self.build_rescue_text(pending)
                    st.session_state["council_rescue_buffer"] = {
                        "round": round_no,
                        "text": text,
                    }
                    st.session_state["council_rescue_running"] = False
                    st.session_state["council_pending_action"] = None
                    st.rerun()

            if cancel_clicked:
                st.session_state["council_pending_action"] = None
                st.rerun()
