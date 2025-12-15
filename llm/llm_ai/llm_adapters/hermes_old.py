# llm/llm_ai/llm_adapters/hermes_old.py
from __future__ import annotations

import os

from llm.llm_ai.llm_adapters.openrouter_chat import OpenRouterChatAdapter


HERMES_MODEL_OLD_DEFAULT = os.getenv(
    "OPENROUTER_HERMES_MODEL",
    # ここは環境変数で上書きされる前提（デフォルトは旧安定版名）
    "nousresearch/hermes-2-pro-mistral",
)


class HermesOldAdapter(OpenRouterChatAdapter):
    """
    旧 Hermes（nousresearch/hermes-2-pro-*）用アダプタ。
    通常運用はこちらを "hermes" として使う想定。
    """

    def __init__(self) -> None:
        super().__init__(
            name="hermes",
            model_id=HERMES_MODEL_OLD_DEFAULT,
        )
        # erotic モードでのやり取りにちょうど良い程度
        self.TARGET_TOKENS = 260
