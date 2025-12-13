from __future__ import annotations
from typing import List, Dict, Any

import streamlit as st

from actors.actor import Actor
from actors.persona.persona_classes.persona_riseria_ja import Persona as RiseriaPersona
from actors.narrator_ai import NarratorAI
from actors.narrator.narrator_manager import NarratorManager
from actors.scene_ai import SceneAI


def get_or_create_riseria_council_manager(player_name: str = "アツシ") -> "CouncilManager":
    key_riseria = "council_manager_riseria"
    key_generic = "council_manager"

    if key_riseria not in st.session_state:
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


class CouncilManager:
    def __init__(
        self,
        partner: Actor | None = None,
        partner_role: str | None = None,
        session_key: str = "council_log",
    ) -> None:
        self.session_key = session_key

        raw_log = st.session_state.get(self.session_key, [])
        self.conversation_log: List[Dict[str, str]] = list(raw_log) if isinstance(raw_log, list) else []

        if partner is None:
            from personas.persona_floria_ja import Persona as FloriaPersona
            partner = Actor("フローリア", FloriaPersona())
            partner_role = "floria"
        else:
            if partner_role is None:
                partner_role = "partner"

        self.partner_role = partner_role
        self.partner = partner
        self.actors = {self.partner_role: self.partner}

        self.state: Dict[str, Any] = {
            "mode": "ongoing",
            "participants": ["player", self.partner_role],
            "last_speaker": self.conversation_log[-1]["role"] if self.conversation_log else None,
            "round0_done": bool(self.conversation_log),
            "special_available": False,
            "special_id": None,
        }

        # world_state を必ず初期化
        SceneAI(state=st.session_state)

        if "narrator_manager" not in st.session_state:
            st.session_state["narrator_manager"] = NarratorManager(state=st.session_state)

        self.narrator_manager = st.session_state["narrator_manager"]
        self.narrator = NarratorAI(
            manager=self.narrator_manager,
            partner_role=self.partner_role,
            partner_name=self.partner.name,
        )

        self._ensure_round0_initialized()

    # -------------------------------------------------

    def _save_log(self) -> None:
        st.session_state[self.session_key] = list(self.conversation_log)

    def _append_log(self, role: str, content: str) -> None:
        safe = (content or "").replace("\n", "<br>")
        self.conversation_log.append({"role": role, "content": safe})
        self.state["last_speaker"] = role
        self._save_log()

    def _ensure_round0_initialized(self) -> None:
        if self.state["round0_done"]:
            return

        try:
            world = SceneAI(state=st.session_state).get_world_state()
            text = self.narrator.generate_round0_opening(
                world_state=world,
                player_profile={},
                floria_state={"mood": "slightly_nervous"},
            ).text
        except Exception:
            text = f"{self.partner.name}は、少し緊張した様子であなたの前に立っている。"

        self._append_log("narrator", text)
        self.state["round0_done"] = True

    # -------------------------------------------------

    def proceed(self, user_text: str) -> str:
        self._append_log("player", user_text)

        reply = ""
        actor = self.actors.get(self.partner_role)
        if actor:
            try:
                reply = actor.speak(self.conversation_log)
            except Exception as e:
                st.error("[Council] Actor.speak failed")
                st.exception(e)
                reply = "……ごめん。今、一瞬だけ考えが途切れたみたい。もう一度話してくれる？"

            self._append_log(self.partner_role, reply)

        return reply

    # -------------------------------------------------

    def render(self) -> None:
        if "council_sending" not in st.session_state:
            st.session_state["council_sending"] = False

        sending = st.session_state["council_sending"]

        st.markdown("## 🗣️ 会談システム")

        for entry in self.conversation_log:
            st.markdown(f"**{entry['role']}**")
            st.markdown(entry["content"], unsafe_allow_html=True)

        user_text = st.text_area("あなたの発言")

        if st.button("送信", disabled=sending):
            if not user_text.strip():
                st.warning("発言を入力してください。")
            else:
                st.session_state["council_sending"] = True
                try:
                    with st.spinner("考えています…"):
                        self.proceed(user_text.strip())
                except Exception as e:
                    st.error("会話処理中にエラーが発生しました")
                    st.exception(e)
                finally:
                    st.session_state["council_sending"] = False
                st.rerun()
