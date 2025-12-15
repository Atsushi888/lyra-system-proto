# llm2/llm_ai/llm_registers/register_gpt4o.py
from __future__ import annotations

from llm2.llm_ai.llm_adapters.gpt4o import GPT4oAdapter


def register_gpt4o(llm_ai: Any) -> None:
    """
    GPT-4o-mini を LLMAI に登録する。
    """
    adapter = GPT4oAdapter()
    llm_ai.register_adapter(adapter)
