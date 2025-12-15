# llm/llm_ai/llm_adapters/hermes_new.py
from __future__ import annotations

from llm.llm_ai.llm_adapters.openrouter_chat import OpenRouterChatAdapter


class HermesNewAdapter(OpenRouterChatAdapter):
    """
    新 Hermes（3/4 系）用アダプタ。

    当面はテスト用として "hermes_new" 名義で扱う。
    """

    def __init__(self) -> None:
        super().__init__(
            name="hermes_new",
            model_id="nousresearch/hermes-4-70b",
        )
        # テスト用途なので少し長めでも許容
        self.TARGET_TOKENS = 320
