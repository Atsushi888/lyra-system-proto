# llm/llm_ai/llm_adapters/llama_unc.py
from __future__ import annotations

import os

from llm.llm_ai.llm_adapters.openrouter_chat import OpenRouterChatAdapter


LLAMA_UNC_MODEL_DEFAULT = os.getenv(
    "OPENROUTER_LLAMA_UNC_MODEL",
    # 実際の OpenRouter モデル ID は環境変数で差し替え可能
    "nousresearch/llama-3.1-70b-instruct:uncensored",
)


class LlamaUncensoredAdapter(OpenRouterChatAdapter):
    """
    Llama 3.1 70B Uncensored 用アダプタ。

    - OpenRouter 上の Uncensored モデルを利用
    - 比較的長文を許容
    """

    def __init__(self) -> None:
        super().__init__(
            name="llama_unc",
            model_id=LLAMA_UNC_MODEL_DEFAULT,
        )
        self.TARGET_TOKENS = 320
