# actors/actor.py

from __future__ import annotations
from typing import List, Dict, Any

import streamlit as st

from personas.persona_floria_ja import Persona
from actors.answer_talker import AnswerTalker


class Actor:
    def __init__(self, name: str, persona: Persona) -> None:
        self.name = name
        self.persona = persona

        # LLMRouterはもう使わない。AnswerTalker内部で自動的にLLMManager/Routerへ接続する
        self.answer_talker = AnswerTalker(persona=self.persona)

    def speak(self, conversation_log: List[Dict[str, str]]) -> str:
        """
        - conversation_log から最新の player 発言を取り出し
        - Persona に messages を作らせ
        - AnswerTalker で LLM パイプラインを回す
        - もし最終返答が空だった場合は、フェイルセーフで「とりあえず何かしゃべる」
        """

        # プレイヤーの最新発言を取得
        user_text = ""
        for entry in reversed(conversation_log):
            if entry.get("role") == "player":
                user_text = entry.get("content", "")
                break

        # ===== DEBUG: 入力ログ =====
        debug_prefix = f"[DEBUG:Actor] {getattr(self, 'name', 'Actor')} "
        st.write(
            f"{debug_prefix}speak() called. "
            f"user_text.len={len(user_text)}, log_len={len(conversation_log)}"
        )

        # Persona に messages を作らせる
        try:
            messages = self.persona.build_messages(user_text)
        except TypeError:
            # もし将来 build_messages(conversation_log) 仕様になっていた場合の保険
            st.write(f"{debug_prefix}build_messages(user_text) TypeError → conversation_log で再試行")
            messages = self.persona.build_messages(conversation_log)

        st.write(f"{debug_prefix}messages built. len={len(messages)}")

        # AnswerTalker によるLLMパイプライン処理
        final_reply = self.answer_talker.speak(messages, user_text=user_text)
        st.write(f"{debug_prefix}AnswerTalker.speak() returned. final_reply.len={len(final_reply)}")

        # ===== フェイルセーフ =====
        safe_reply = (final_reply or "").strip()
        if not safe_reply:
            st.warning(
                f"{debug_prefix}final_reply is empty. Fallback message will be used."
            )
            # 会話が完全に死ぬのを防ぐための暫定セリフ
            safe_reply = (
                "あっ……ごめんなさい、ちょっと考え込んじゃってました。"
                "もう一度だけ、ゆっくり聞かせていただけますか？"
            )

        return safe_reply
