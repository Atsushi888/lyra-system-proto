# llm2/llm_ai/llm_registers/register_llama_unc.py
from __future__ import annotations

from llm2.llm_ai.llm_adapters.llama_unc import LlamaUncensoredAdapter


def register_llama_unc(llm_ai: Any) -> None:
    """
    Llama 3.1 70B Uncensored を LLMAI に登録する。
    """
    adapter = LlamaUncensoredAdapter()
    llm_ai.register_adapter(adapter)
