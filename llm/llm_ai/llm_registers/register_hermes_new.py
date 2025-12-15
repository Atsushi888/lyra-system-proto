# llm2/llm_ai/llm_registers/register_hermes_new.py
from __future__ import annotations

from llm.llm_ai.llm_adapters.hermes_new import HermesNewAdapter


def register_hermes_new(llm_ai: Any) -> None:
    """
    新 Hermes（hermes-4 系）を LLMAI に登録する。
    """
    adapter = HermesNewAdapter()
    llm_ai.register_adapter(adapter)
