# actors/actor.py

from __future__ import annotations
from typing import List, Dict, Any

from personas.persona_floria_ja import Persona
from llm.llm_router import LLMRouter
from actors.answer_talker import AnswerTalker


class Actor:
    def __init__(self, name: str, persona: Persona, router: LLMRouter | None = None) -> None:
        self.name = name
        self.persona = persona
        self.router = router or LLMRouter()
        self.answer_talker = AnswerTalker()

    def speak(self, conversation_log: List[Dict[str, str]]) -> str:
        # 最新のプレイヤー発言を拾う
        user_text = ""
        for entry in reversed(conversation_log):
            if entry.get("role") == "player":
                user_text = entry.get("content", "")
                break

        # Persona で messages を組み立てる
        messages = self.persona.build_messages(user_text)

        # 従来どおり GPT-4o に一発投げる
        result = self.router.call_gpt4o(messages)

        # reply_text 抽出
        if isinstance(result, tuple):
            reply_text = result[0]
        else:
            reply_text = result

        messages = self.persona.build_messages(user_text)
        final_reply = self.answer_talker.speak(messages, user_text=user_text)
        # return final_reply
        return reply_text
