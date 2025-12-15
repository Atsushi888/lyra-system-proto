# llm2/llm_ai/llm_registers/register_gemini.py
from __future__ import annotations

from llm.llm_ai.llm_adapters.gemini import GeminiAdapter


def register_gemini(llm_ai: Any) -> None:
    """
    Gemini を LLMAI に登録する。
    """
    adapter = GeminiAdapter()
    llm_ai.register_adapter(adapter)
