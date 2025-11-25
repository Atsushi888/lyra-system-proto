# actors/llm_adapters/gemini_ai.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from actors.llm_ai import LLMAI
from llm.llm_adapter import GeminiAdapter  # 既存 adapter を再利用


class GeminiAI(LLMAI):
    """
    Google Gemini 2.0 用 LLMAI。
    すべての judge_mode で参加。
    """

    def __init__(self) -> None:
        super().__init__(
            name="gemini",
            family="gemini-2.0",
            modes=["all"],
            enabled=True,
        )
        self._adapter = GeminiAdapter()

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        return self._adapter.call(messages=messages, **kwargs)
