# llm/llm_manager_factory.py

from __future__ import annotations

from llm.llm_manager import LLMManager


def get_llm_manager(persona_id: str = "default") -> LLMManager:
    """
    互換性維持用の薄いラッパ。

    既存コードはそのまま:
        mgr = get_llm_manager("floria_ja")

    実体は LLMManager.get() に集約される。
    """
    return LLMManager.get(persona_id)
