# actors/actor.py

from __future__ import annotations
from typing import List, Dict

from llm.llm_router import call_gpt4o


class Actor:
    """
    会談システム用のシンプルな「話し手」クラス。

    - name: 画面表示用の名前（例: "フローリア"）
    - persona: いずれ使うかもしれないので受け取っておくだけ（今は system プロンプトを内部で固定）
    """

    def __init__(self, name: str, persona: object | None = None) -> None:
        self.name = name
        self.persona = persona

    # --- LLM に渡す messages を組み立て ---
    def _build_messages(self, conversation_log: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        いまのところ「直近のプレイヤー発言」だけを user に渡す。
        前後文脈などは、後でちゃんと設計するときに拡張する。
        """
        last_user_text = ""
        for m in reversed(conversation_log):
            if m.get("role") == "player":
                last_user_text = m.get("content", "")
                break

        # HTML の <br> を改行に戻してから LLM に渡す
        last_user_text = last_user_text.replace("<br>", "\n")

        system_prompt = (
            "あなたは『フローリア』という名の、水と氷の精霊の乙女です。"
            "プレイヤーの恋人として、優しく、穏やかに、日本語で会話してください。\n"
            "一人称は「わたし」。\n"
            "会話はロマンチックで感情豊かに。ただし露骨な性描写は用いず、"
            "ソフトで甘い雰囲気を大事にしてください。"
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": last_user_text or "プレイヤーからの言葉に続けて、自然に会話してください。"},
        ]
        return messages

    # --- 会話を 1 ターン進める ---
    def speak(self, conversation_log: List[Dict[str, str]]) -> str:
        messages = self._build_messages(conversation_log)
        text, meta = call_gpt4o(
            messages=messages,
            temperature=0.7,
            max_tokens=600,
        )
        return text
