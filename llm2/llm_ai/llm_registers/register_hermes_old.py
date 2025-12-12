# llm2/llm_ai/llm_registers/register_hermes_old.py
from __future__ import annotations

from llm2.llm_ai.llm_adapters.hermes_old import HermesOldAdapter


def register_hermes_old(llm_ai: Any) -> None:
    """
    旧 Hermes（hermes-2-pro 系）を LLMAI に登録する。
    """
    adapter = HermesOldAdapter()
    llm_ai.register_adapter(adapter)
