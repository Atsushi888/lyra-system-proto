# actors/council_manager.py
from __future__ import annotations
from typing import List, Dict, Any

import streamlit as st

from actors.actor import Actor
from personas.persona_floria_ja import Persona as FloriaPersona
from actors.persona.persona_classes.persona_riseria_ja import Persona as RiseriaPersona
from actors.narrator_ai import NarratorAI
from actors.narrator.narrator_manager import NarratorManager
from actors.scene_ai import SceneAI


# ==========================================================
# CouncilManager を取得するヘルパ
# ==========================================================

def get_or_create_council_actor() -> Actor:
    """
    互換性のためのヘルパ。
    既存コードでは「フローリア前提」で Actor を取得しているので、
    ここは従来どおりフローリア Actor を返す。
    """
    actor_key = "council_actor"

    if actor_key not in st.session_state:
        st.session_state[actor_key] = Actor(
            name="フローリア",
            persona=FloriaPersona(),
        )

    return st.session_state[actor_key]


def get_or_create_floria_council_manager() -> "CouncilManager":
    """
    既存フローリア用の CouncilManager をセッションから取得（なければ作成）。
    """
    key = "council_manager_floria"

    if key not in st.session_state:
        floria_actor = Actor(name="フローリア", persona=FloriaPersona())
        st.session_state[key] = CouncilManager(
            partner=floria_actor,
            partner_role="floria",
            session_key="council_log_floria",
        )

    return st.session_state[key]


def get_or_create_riseria_council_manager(player_name: str = "アツシ") -> "CouncilManager":
    """
    リセリアとの会話用 CouncilManager をセッションから取得（なければ作成）。

    - Persona は actors/persona/persona_datas/elf_riseria_da_silva_ja.json を元に構築
    - Actor.name には Persona.display_name（通常「リセリア・ダ・シルヴァ」）を使用
    - partner_role は "riseria"
    """
    key = "council_manager_riseria"

    if key not in st.session_state:
        st.write(f"[DEBUG:Council] create CouncilManager for Riseria (player_name={player_name})")

        riseria_persona = RiseriaPersona(player_name=player_name)
        riseria_actor = Actor(
            name=riseria_persona.display_name,
            persona=riseria_persona,
        )
        st.session_state[key] = CouncilManager(
            partner=riseria_actor,
            partner_role="riseria",
            session_key="council_log_riseria",
        )

    return st.session_state[key]


# ==========================================================
# CouncilManager 本体
# ==========================================================

class CouncilManager:
    """
    会談システムのロジック ＋ 画面描画（β）。

    - デフォルトではフローリアとの会話になる。
    - partner / partner_role を指定することで、会話相手を差し替え可能。
    """

    def __init__(
        self,
        partner: Actor | None = None,
        partner_role: str | None = None,
        session_key: str = "council_log",
    ) -> None:
        st.write(
            f"[DEBUG:Council] CouncilManager.__init__ partner="
            f"{getattr(partner, 'name', 'None')}, partner_role={partner_role}"
        )

        self.session_key = session_key

        # ===== 会話ログ（まずセッションからロード） =====
        raw_log = st.session_state.get(self.session_key, [])
        self.conversation_log: List[Dict[str, str]] = list(raw_log) if isinstance(raw_log, list) else []

        # ===== 会話相手（デフォルトはフローリア） =====
        if partner is None:
            partner = Actor("フローリア", FloriaPersona())
            partner_role = "floria"
        else:
            if partner_role is None:
                partner_role = "partner"

        self.partner_role: str = partner_role
        self.partner: Actor = partner

        # いまは 1on1（＋ナレーション）想定
        self.actors: Dict[str, Actor] = {
            self.partner_role: self.partner
        }

        # ===== 状態 =====
        # round0_done は「ログに Round0 が差し込まれているか」で判定
        self.state: Dict[str, Any] = {
            "mode": "ongoing",
            "participants": ["player", self.partner_role],
            "last_speaker": self.conversation_log[-1]["role"] if self.conversation_log else None,
            "round0_done": bool(self.conversation_log),
            "special_available": False,
            "special_id": None,
        }

        st.write(
            f"[DEBUG:Council] load conversation_log from session: "
            f"len={len(self.conversation_log)} (key={self.session_key})"
        )

        # world_state を必ず初期化しておく
        SceneAI(state=st.session_state)  # __init__ の中で ensure_world_initialized が走る
        st.write("[DEBUG:Council] initialize SceneAI world_state (ensure_world_initialized)")

        # NarratorManager / NarratorAI
        if "narrator_manager" not in st.session_state:
            st.session_state["narrator_manager"] = NarratorManager(state=st.session_state)
        self.narrator_manager: NarratorManager = st.session_state["narrator_manager"]

        st.write("[DEBUG:Council] create NarratorManager / NarratorAI")
        self.narrator = NarratorAI(
            manager=self.narrator_manager,
            partner_role=self.partner_role,
            partner_name=getattr(self.partner, "name", self.partner_role),
        )

        # Round0 を 1 回だけ差し込む
        self._ensure_round0_initialized()

    # ------------------------------------------------------
    # world_state 関連ヘルパ
    # ------------------------------------------------------
    def _get_world_snapshot(self) -> Dict[str, Any]:
        llm_meta = st.session_state.get("llm_meta", {})
        world = llm_meta.get("world") or {}
        if not world:
            scene_ai = SceneAI(state=st.session_state)
            world = scene_ai.get_world_state()
        return world

    def _build_narrator_world_state(self) -> Dict[str, Any]:
        """
        NarratorAI に渡す world_state を llm_meta["world"] から構成する。
        """
        world = self._get_world_snapshot()
        locs = world.get("locations", {})
        t = world.get("time", {})

        location_name = locs.get("player") or "通学路"
        time_of_day = t.get("slot", "morning")
        weather = world.get("weather", "clear")

        return {
            "location_name": location_name,
            "time_of_day": time_of_day,
            "weather": weather,
        }

    # ------------------------------------------------------
    # ログ操作
    # ------------------------------------------------------
    def _save_log_to_session(self) -> None:
        st.session_state[self.session_key] = list(self.conversation_log)
        st.write(
            f"[DEBUG:Council] save conversation_log to session: "
            f"len={len(self.conversation_log)} (key={self.session_key})"
        )

    def _append_log(self, role: str, content: str) -> None:
        safe = (content or "").replace("\n", "<br>")
        self.conversation_log.append({"role": role, "content": safe})
        self.state["last_speaker"] = role
        self._save_log_to_session()
        st.write(
            f"[DEBUG:Council] _append_log role={role}, len(log)={len(self.conversation_log)}, "
            f"preview='{safe[:40]}'"
        )

    def _ensure_round0_initialized(self) -> None:
        """
        会談開始時のナレーション（Round0）を一度だけ差し込む。
        相手キャラクターは self.partner を前提にしている。
        """
        if self.state.get("round0_done", False):
            st.write("[DEBUG:Council] round0 already done, skip.")
            return

        st.write("[DEBUG:Council] generate Round0 narration")
        world_state = self._build_narrator_world_state()
        player_profile: Dict[str, Any] = {}

        # 将来自キャラの状態も world_state から拾って拡張可能
        partner_state = {"mood": "slightly_nervous"}

        try:
            line = self.narrator.generate_round0_opening(
                world_state=world_state,
                player_profile=player_profile,
                floria_state=partner_state,  # NarratorAI 側の引数名は現状のまま
            )
            text = (getattr(line, "text", None) or "").strip()
        except Exception as e:
            text = ""
            st.warning(f"[DEBUG:Council] Round0 narration error: {e}")

        if not text:
            st.warning("[DEBUG:Council] Round0 narration was empty. Used fallback text.")
            text = f"{getattr(self.partner, 'name', 'その子')}は、どこかそわそわした様子であなたの前に立っている。"

        self._append_log("narrator", text)
        self.state["round0_done"] = True
        st.write(
            f"[DEBUG:Council] round0_done set True, log_len={len(self.conversation_log)}"
        )

    # ------------------------------------------------------
    # 公開 API（ロジック）
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

        self._save_log_to_session()
        self._ensure_round0_initialized()

    def get_log(self) -> List[Dict[str, str]]:
        return list(self.conversation_log)

    def get_status(self) -> Dict[str, Any]:
        round_ = len(self.conversation_log) + 1
        world = self._get_world_snapshot()
        locs = world.get("locations", {})
        t = world.get("time", {})

        return {
            "round": round_,
            "speaker": "player",
            "mode": self.state.get("mode", "ongoing"),
            "participants": self.state.get("participants", ["player", self.partner_role]),
            "last_speaker": self.state.get("last_speaker"),
            "special_available": self.state.get("special_available", False),
            "world": {
                "player_location": locs.get("player"),
                # TODO: location は将来 partner_role ごとに持たせる
                "floria_location": locs.get("floria"),
                "time_slot": t.get("slot"),
                "time_str": t.get("time_str"),
            },
        }

    def proceed(self, user_text: str) -> str:
        """
        プレイヤー発言 user_text をログに追加し、
        現在の会話相手 Actor に発言させて、その内容を返す。
        """
        st.write(f"[DEBUG:Council] proceed() user_text='{user_text[:40]}'")
        self._append_log("player", user_text)

        reply = ""
        actor = self.actors.get(self.partner_role)
        if actor is not None:
            st.write(
                f"[DEBUG:Council] call Actor.speak() for partner_role={self.partner_role}, "
                f"partner_name={getattr(self.partner, 'name', self.partner_role)}"
            )
            reply = actor.speak(self.conversation_log)
            self._append_log(self.partner_role, reply)

        return reply

    # ------------------------------------------------------
    # 救済アクション（ロジック）
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
    # 画面描画（UI）
    # ------------------------------------------------------
    def render(self) -> None:
        """
        以前の council_manager.py に入っていた UI 部分を
        そのまま保持した render()。
        """
        if "council_sending" not in st.session_state:
            st.session_state["council_sending"] = False
        if "council_pending_action" not in st.session_state:
            st.session_state["council_pending_action"] = None
        if "council_rescue_running" not in st.session_state:
            st.session_state["council_rescue_running"] = False

        sending: bool = st.session_state["council_sending"]

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
                    name = "プレイヤー"
                elif role == "narrator":
                    name = "ナレーション"
                elif role == self.partner_role:
                    name = getattr(self.partner, "name", self.partner_role)
                else:
                    name = role or "？"

                st.markdown(f"**[{idx}] {name}**")
                st.markdown(text, unsafe_allow_html=True)
                st.markdown("---")

        # ---- サイドバー：会談ステータス ----
        with st.sidebar.expander("📊 会談ステータス", expanded=True):
            st.write(f"ラウンド: {status.get('round')}")
            st.write(f"話者: {status.get('speaker')}")
            st.write(f"モード: {status.get('mode')}")
            participants = status.get("participants") or []
            if participants:
                label_map = {
                    "player": "プレイヤー",
                    self.partner_role: getattr(self.partner, "name", self.partner_role),
                }
                labels = [label_map.get(p, p) for p in participants]
                st.write("参加者: " + " / ".join(labels))
            last = status.get("last_speaker")
            if last:
                st.write(f"最後の話者: {last}")
            st.write(f"スペシャル選択可: {status.get('special_available')}")

            st.markdown("---")
            st.write("**現在の世界情報**")
            st.write(f"プレイヤー位置: {world_info.get('player_location')}")
            st.write(f"フローリア位置: {world_info.get('floria_location')}")
            st.write(f"時間帯: {world_info.get('time_slot')}")
            st.write(f"時刻: {world_info.get('time_str')}")

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
            placeholder=f"ここに{getattr(self.partner, 'name', '相手キャラクター')}への発言を書いてください。",
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

        if send_clicked:
            cleaned = (user_text or "").strip()
            if not cleaned:
                st.warning("発言を入力してください。")
            else:
                if st.session_state["council_sending"]:
                    st.info("いま処理中です。少し待ってから再度お試しください。")
                else:
                    st.session_state["council_sending"] = True
                    with st.spinner(f"{getattr(self.partner, 'name', '相手')}は少し考えています…"):
                        self.proceed(cleaned)
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

        pending = st.session_state.get("council_pending_action")
        if pending:
            if pending == "wait":
                msg = "このターンは何も行動せず、様子を見ます。よろしいですか？"
            elif pending == "look_person":
                msg = "隣にいる相手の様子をうかがいます。よろしいですか？"
            elif pending == "scan_area":
                msg = "周囲の様子を見回します。よろしいですか？"
            elif pending == "special":
                world_state = self._build_narrator_world_state()
                special_id = self.state.get("special_id") or "unknown_special"
                title, _ = self.narrator.make_special_title_and_choice(
                    special_id,
                    world_state=world_state,
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
