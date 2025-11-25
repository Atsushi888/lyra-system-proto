from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from actors.llm_ai import LLMAI
from llm.llm_adapter import HermesOldAdapter  # 既存 adapter を再利用


class HermesOldAI(LLMAI):
    """
    旧 Hermes（安定版）用 LLMAI。

    デバッグ優先版:
      - judge_mode に関わらず常に回答対象にするため、modes=["all"] にしてある。
      - 実際の文体（えっち寄り etc）は、上位のプロンプト側でコントロール。
    """

    def __init__(self) -> None:
        super().__init__(
            name="hermes",
            family="hermes-2",
            # ★ ここを ["erotic"] → ["all"] に変更
            #   → LLMAI.should_answer() が常に True を返す（enabled=True の限り）
            modes=["all"],
            enabled=True,
        )
        self._adapter = HermesOldAdapter()

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        # そのまま既存 adapter に委譲
        return self._adapter.call(messages=messages, **kwargs)
