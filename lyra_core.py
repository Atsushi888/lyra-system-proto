# lyra_core.py

from typing import Dict, List, Tuple
from personas.persona_floria_ja import get_persona
from multi_ai import AIResponder

class LyraCore:
    def __init__(self, conversation):
        persona = get_persona()
        self.conversation = conversation  # 使うなら保持

        self.responder_4o = AIResponder(
            system_prompt=persona.system_prompt,
            style_hint=persona.style_hint,
        )
        self.responder_hermes = AIResponder(
            system_prompt=persona.system_prompt,
            style_hint=persona.style_hint,
        )
        
    def proceed_turn(self, user_text: str, state) -> Tuple[List[Dict[str, str]], Dict]:
        messages = state.get("messages", [])
        messages.append({"role": "user", "content": user_text})

        # 各モデルに同じメッセージを渡す
        reply_4o, meta_4o = self.responder_4o.reply(messages)
        reply_hermes, meta_hermes = self.responder_hermes.reply(messages)

        # メイン出力（例: GPT-4o版）をログに追加
        messages.append({"role": "assistant", "content": reply_4o})

        # 比較情報をまとめて返す
        meta = {
            "gpt4o": {"reply": reply_4o, "meta": meta_4o},
            "hermes": {"reply": reply_hermes, "meta": meta_hermes},
        }

        return messages, meta
