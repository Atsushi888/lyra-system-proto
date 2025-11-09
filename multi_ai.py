# multi_ai.py

# from typing import Dict, List, Tuple
# from conversation_engine import LLMConversation


# class AIResponder:
#     """1モデルぶんの薄いラッパ。"""

#     def __init__(self, system_prompt: str, style_hint=None):
#         self.conversation = LLMConversation(
#             system_prompt=system_prompt,
#             temperature=0.7,
#             max_tokens=800,
#             style_hint=style_hint,
#         )

#     def reply(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict]:
#         # ここは、あなたの LLMConversation のメソッド名に合わせる
#         # 例：もし generate(messages) があるならそれを使う
#         text, meta = self.conversation.generate(messages)
#         return text, meta


from typing import Dict, List, Tuple
from conversation_engine import LLMConversation

class AIResponder:
    def __init__(self, system_prompt: str, style_hint=None):
        self.conversation = LLMConversation(
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=800,
            style_hint=style_hint,
        )

    def reply(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict]:
        text, meta = self.conversation.generate_reply(messages)
        return text, meta
