# llm/llm_router.py

from __future__ import annotations
import os
from typing import List, Dict, Any, Tuple

from openai import OpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


class LLMRouter:
    """
    GPT-4o / GPT-5.1 / Hermes などを呼び分ける中心クラス。
    まずは call_gpt4o だけ使う。
    """

    def __init__(self, persona: Any = None):
        self.persona = persona

    # ============================
    # GPT-4o 呼び出し
    # ============================
    def call_gpt4o(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        GPT-4o を呼び出す最小ユニット。
        messages は Streamlit の conversation_log に準拠。
        """
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

        reply_text = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }

        return reply_text, usage
