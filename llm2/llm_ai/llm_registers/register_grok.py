from __future__ import annotations

from typing import Any

from llm2.llm_ai.llm_adapters.grok import GrokAdapter


def register_grok(llm_ai: Any) -> None:
    """
    Grok を LLMAI に登録する。
    """
    adapter = GrokAdapter()
    llm_ai.register_adapter(adapter)
