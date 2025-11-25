# actors/llm_adapters/gpt4o_ai.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from actors.llm_ai import LLMAI
from llm.llm_adapter import GPT4oAdapter  # 既存 adapter を再利用


class GPT4oAI(LLMAI):
    """
    gpt-4o 用 LLMAI サブクラス。
    現在は multi-LLM からは参加させない（enabled=False, modes=["none"]）。
    将来的に再活用したくなったときのために枠だけ残しておく。
    """

    def __init__(self) -> None:
        # modes=["none"] + enabled=False → どの judge_mode でも参加しない
        super().__init__(
            name="gpt4o",
            family="gpt-4o",
            modes=["none"],
            enabled=False,
        )
        self._adapter = GPT4oAdapter()

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        # 万一直接呼ばれたら、adapter 経由でそのまま実行
        return self._adapter.call(messages=messages, **kwargs)
