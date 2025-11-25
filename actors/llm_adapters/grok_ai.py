# actors/llm_adapters/grok_ai.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from actors.llm_ai import LLMAI
from llm.llm_adapter import GrokAdapter  # 既存 adapter を再利用


class GrokAI(LLMAI):
    """
    xAI Grok 用 LLMAI。
    すべての judge_mode で参加させる。
    """

    def __init__(self) -> None:
        super().__init__(
            name="grok",
            family="grok-2",
            modes=["all"],
            enabled=True,
        )
        self._adapter = GrokAdapter()

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        return self._adapter.call(messages=messages, **kwargs)
