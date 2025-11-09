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
            model_name=model_name,
        )
        self.model_name = model_name

    def reply(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict]:
        text, meta = self.conversation.generate(messages)
        return text, meta
