# actors/llm_adapters/hermes_old_ai.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from actors.llm_ai import LLMAI
from llm.llm_adapter import HermesOldAdapter  # 既存 adapter を再利用


class HermesOldAI(LLMAI):
    """
    旧 Hermes（安定版）用 LLMAI。
    erotic モード専用参加とする。
    """

    def __init__(self) -> None:
        super().__init__(
            name="hermes",
            family="hermes-2",
            modes=["erotic"],   # erotic のときだけ should_answer=True
            enabled=True,
        )
        self._adapter = HermesOldAdapter()

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        return self._adapter.call(messages=messages, **kwargs)
