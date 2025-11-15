# actors/actor.py

from typing import List, Dict, Any
from llm.llm_router import LLMRouter


class Actor:
    """
    会談システムの参加者（AIキャラ）。
    persona と LLMRouter を内側に持つ。
    """

    def __init__(self, name: str, persona: Any):
        self.name = name
        self.persona = persona
        self.router = LLMRouter(persona=persona)

    def speak(self, messages: List[Dict[str, str]]) -> str:
        """
        過去ログを渡して GPT-4o に返事させる最小構成。
        ここから multi-AI / judge / composer を後で入れる。
        """
        reply_text, usage = self.router.call_gpt4o(
            messages=messages,
            temperature=getattr(self.persona, "temperature", 0.7),
            max_tokens=800,
        )
        return reply_text
