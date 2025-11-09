# multi_ai.py

from typing import Dict, List, Tuple
from conversation_engine import LLMConversation


class AIResponder:
    """1モデルぶんのラッパ。どのLLMでも統一して使えるように。"""

    def __init__(self, model_name: str, system_prompt: str, style_hint=None):
        self.conversation = LLMConversation(
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=800,
            style_hint=style_hint,
        )

    def reply(self, messages):
        text, meta = self.conversation.generate(messages)
        return text, meta
