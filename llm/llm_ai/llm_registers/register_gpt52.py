from __future__ import annotations

from typing import Any
from llm.llm_ai.llm_adapters.openai_chat import OpenAIChatAdapter


def register_gpt52(llm_ai: Any) -> None:
    """
    GPT-5.2 Chat を LLMAI に登録する。
    model_id は OpenAI公式の "gpt-5.2-chat-latest" を使う。  #  [oai_citation:1‡OpenAI Platform](https://platform.openai.com/docs/models/gpt-5.2-chat-latest?utm_source=chatgpt.com)
    """
    adapter = OpenAIChatAdapter(
        name="gpt52",
        model_id="gpt-5.2-chat-latest",
    )
    llm_ai.register_adapter(adapter)
