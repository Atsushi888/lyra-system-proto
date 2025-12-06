from __future__ import annotations
from typing import List, Dict, Any

import streamlit as st

from actors.actor import Actor
from personas.persona_floria_ja import Persona as FloriaPersona
from actors.persona.persona_classes.persona_riseria_ja import Persona as RiseriaPersona
from actors.narrator_ai import NarratorAI
from actors.narrator.narrator_manager import NarratorManager
from actors.scene_ai import SceneAI


# 会話ログをセッションに保存するキー（相手ごとに分離）
SESSION_KEY_LOG_RISERIA = "council_log_riseria"
SESSION_KEY_LOG_FLORIA = "council_log_floria"


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


# ==========================================================
# 追加: フローリア用 / リセリア用 CouncilManager ヘルパ
# ==========================================================

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
            session_log_key=SESSION_KEY_LOG_FLORIA,
        )

    return st.session_state[key]


def get_or_create_riseria_council_manager(player_name: str = "アツシ") -> "CouncilManager":
    """
    リセリアとの会話用 CouncilManager をセッションから取得（なければ作成）。
    """
    key = "council_manager_riseria"

    if key not in st.session_state:
        riseria_persona = RiseriaPersona(player_name=player_name)
        riseria_actor = Actor(
            name=riseria_persona.display_name,
            persona=riseria_persona,
        )
        st.session_state[key] = CouncilManager(
            partner=riseria_actor,
            partner_role="riseria",
            session_log_key=SESSION_KEY_LOG_RISERIA,
        )

    return st.session_state[key]


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
        session_log_key: str | None = None,
    ) -> None:
        self.session_log_key = session_log_key or "council_log_generic"

        # 会話ログ（まずはセッションから復元）
        restored = st.session_state.get(self.session_log_key)
        if isinstance(restored, list):
            self.conversation_log: List[Dict[str, str]] = list(restored)
            st.write(
                f"[DEBUG:Council] load conversation_log from session: "
                f"len={len(self.conversation_log)} (key={self.session_log_key})"
            )
        else:
            self.conversation_log = []

        # ===== 会話相手（デフォルトはフローリア） =====
        if partner is None:
            partner = Actor("フローリア", FloriaPersona())
            partner_role = "floria"
        else:
            if partner_role is None:
                partner_role = "partner"

        self.partner_role: str = partner_role
        self.partner: Actor = partner

        st.write(
            f"[DEBUG:Council] CouncilManager.__init__ partner="
            f"{getattr(self.partner, 'name', '???')}, partner_role={self.partner_role}"
        )

        # いまは 1on1（＋ナレーション）想定
        self.actors: Dict[str, Actor] = {
            self.partner_role: self.partner
        }

        self.state: Dict[str, Any] = {
            "mode": "ongoing",
            "participants": ["player", self.partner_role],
            "last_speaker": None,
            "round0_done": len(self.conversation_log) > 0,
            "special_available": False,
            "special_id": None,
        }

        # world_state を必ず初期化しておく
        SceneAI(state=st.session_state)  # __init__ の中で ensure_world_initialized が走る
        st.write("[DEBUG:Council] initialize SceneAI world_state (ensure_world_initialized)")

        # NarratorManager / NarratorAI
        if "narrator_manager" not in st.session_state:
            st.session_state["narrator_manager"] = NarratorManager(state=st.session_state)
        self.narrator_manager: NarratorManager = st.session_state["narrator_manager"]

        # ★ ここで partner_role / partner_name を渡す
        self.narrator = NarratorAI(
            manager=self.narrator_manager,
            partner_role=self.partner_role,
            partner_name=getattr(self.partner, "name", self.partner_role),
        )
        st.write(
            f"[DEBUG:Council] create NarratorManager / NarratorAI "
            f"(partner={self.partner.name}, role={self.partner_role})"
        )

        # Round0 を 1 回だけ差し込む
        self._ensure_round0_initialized()

    # ===== world_state 関連ヘルパ =====
    def _get_world_snapshot(self) -> Dict[str, Any]:
        llm_meta = st.session_state.get("llm_meta", {})
        world = llm_meta.get("world") or {}
        if not world:
            scene_ai = SceneAI(state=st.session_state)
            world = scene_ai.get_world_state()
        return world

    def _build_narrator_world_state(self) -> Dict[str, Any]:
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

    # ===== ログ操作 =====
    def _append_log(self, role: str, content: str) -> None:
        safe = (content or "").replace("\n", "<br>")
        self.conversation_log.append({"role": role, "content": safe})
        self.state["last_speaker"] = role

        st.write(
            f"[DEBUG:Council] _append_log role={role}, len(log)={len(self.conversation_log)}, "
            f"preview='{safe[:40]}'"
        )

        # 追記ごとにセッションへ保存
        st.session_state[self.session_log_key] = list(self.conversation_log)
        st.write(
            f"[DEBUG:Council] save conversation_log to session: "
            f"len={len(self.conversation_log)} (key={self.session_log_key})"
        )

    def _ensure_round0_initialized(self) -> None:
        """
        会談開始時のナレーション（Round0）を一度だけ差し込む。
        相手キャラクターは self.partner を前提にしている。
        """
        if self.conversation_log:
            st.write(
                f"[DEBUG:Council] round0 already done (log_len={len(self.conversation_log)})"
            )
            return

        st.write("[DEBUG:Council] generate Round0 narration")

        world_state = self._build_narrator_world_state()
        player_profile: Dict[str, Any] = {}
        partner_state = {"mood": "slightly_nervous"}

        line = self.narrator.generate_round0_opening(
            world_state=world_state,
            player_profile=player_profile,
            floria_state=partner_state,  # NarratorAI 側の引数名は現状のまま
        )

        # NarratorAI が空文字を返してきた場合のフェイルセーフ
        text = getattr(line, "text", "") if line is not None else ""
        if not (text or "").strip():
            text = (
                f"{getattr(self.partner, 'name', '相手キャラクター')}は、"
                "どこかそわそわした様子であなたの前に立っている。"
            )
            st.warning("[DEBUG:Council] Round0 narration was empty. Used fallback text.")

        self._append_log("narrator", text)
        self.state["round0_done"] = True

        st.write(
            f"[DEBUG:Council] round0_done set True, log_len={len(self.conversation_log)}"
        )

    # ===== 公開 API =====
    def reset(self) -> None:
        self.conversation_log.clear()
        self.state["mode"] = "ongoing"
        self.state["last_speaker"] = None
        self.state["round0_done"] = False
        self.state["special_available"] = False
        self.state["special_id"] = None

        st.session_state.pop("council_rescue_buffer", None)
        st.session_state.pop("council_pending_action", None)
        st.session_state.pop(self.session_log_key, None)

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
        st.write(f"[DEBUG:Council] proceed() user_text='{user_text[:32]}', len(log)={len(self.conversation_log)}")
        self._append_log("player", user_text)

        reply = ""
        actor = self.actors.get(self.partner_role)
        if actor is not None:
            st.write(
                f"[DEBUG:Council] call Actor.speak() for partner={self.partner_role}, "
                f"partner_name={getattr(self.partner, 'name', self.partner_role)}"
            )
            reply = actor.speak(self.conversation_log)
            self._append_log(self.partner_role, reply)

        return reply

    # （以下の render() / rescue 関連はそのまま。省略）
