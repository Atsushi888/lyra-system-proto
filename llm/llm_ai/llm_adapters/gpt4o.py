# llm2/llm_ai/llm_adapters/gpt4o.py
from __future__ import annotations

from llm.llm_ai.llm_adapters.openai_chat import OpenAIChatAdapter


class GPT4oAdapter(OpenAIChatAdapter):
    """
    gpt-4o-mini 用アダプタ。

    現状は補助用途・テスト用途想定。
    トークン数は明示的に制御しない。
    """

    def __init__(self) -> None:
        super().__init__(name="gpt4o", model_id="gpt-4o-mini")
        self.TARGET_TOKENS = None
