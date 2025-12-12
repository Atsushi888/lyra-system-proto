# llm2/llm_ai/llm_registers/register_gpt51.py
from __future__ import annotations

from llm2.llm_ai.llm_adapters.gpt51 import GPT51Adapter


def register_gpt51(llm_ai: Any) -> None:
    """
    GPT-5.1 を LLMAI に登録する。
    """
    adapter = GPT51Adapter()
    llm_ai.register_adapter(adapter)
