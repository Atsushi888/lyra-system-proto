# actors/actor.py

from __future__ import annotations
from typing import List, Dict, Any

from personas.persona_floria_ja import Persona
from actors.answer_talker import AnswerTalker


class Actor:
    def __init__(self, name: str, persona: Persona) -> None:
        self.name = name
        self.persona = persona

        # LLMRouterはもう使わない。AnswerTalker内部で自動的にLLMManager/Routerへ接続する
        self.answer_talker = AnswerTalker(persona=self.persona)

    def speak(self, conversation_log: List[Dict[str, str]]) -> str:
        # プレイヤーの最新発言を取得
        user_text = ""
        for entry in reversed(conversation_log):
            if entry.get("role") == "player":
                user_text = entry.get("content", "")
                break

        # Persona に messages を作らせる
        messages = self.persona.build_messages(user_text)

        # AnswerTalker によるLLMパイプライン処理
        final_reply = self.answer_talker.speak(messages, user_text=user_text)
        return final_reply
