# actors/llm_adapters/hermes_new_ai.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from actors.llm_ai import LLMAI
from llm.llm_adapter import HermesNewAdapter  # 既存 adapter を再利用


class HermesNewAI(LLMAI):
    """
    新 Hermes（3/4 系）用 LLMAI。
    デフォルトでは無効。テストしたくなったら enabled を True にして使う。
    """

    def __init__(self) -> None:
        super().__init__(
            name="hermes_new",
            family="hermes-4",
            modes=["all"],      # 有効化された場合は全モード参加
            enabled=False,      # ★ ここデフォルト無効
        )
        self._adapter = HermesNewAdapter()

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        return self._adapter.call(messages=messages, **kwargs)
