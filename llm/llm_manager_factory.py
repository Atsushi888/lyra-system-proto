from __future__ import annotations

from llm.llm_manager import LLMManager


def get_llm_manager(persona_id: str = "default") -> LLMManager:
    """
    互換性維持用の薄いラッパ。

    以前はこのモジュール側にロジックを置いていたが、
    現在は LLMManager.get_or_create() が本体。
    """
    return LLMManager.get_or_create(persona_id=persona_id)
