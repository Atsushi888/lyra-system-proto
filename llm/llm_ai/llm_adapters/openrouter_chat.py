# llm2/llm_ai/llm_adapters/openrouter_chat.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import logging
import requests

from llm2.llm_ai.llm_adapters.base import BaseLLMAdapter
from llm2.llm_ai.llm_adapters.utils import (
    split_text_and_usage_from_dict,
)

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)


class OpenRouterChatAdapter(BaseLLMAdapter):
    """
    OpenRouter ChatCompletion 系（Hermes / Llama Uncensored など）の共通アダプタ。

    - requests で直接 OpenRouter API を叩く
    - max_tokens / TARGET_TOKENS の扱いを内部で統一
    """

    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        env_key: str = "OPENROUTER_API_KEY",
    ) -> None:
        self.name = name
        self.model_id = model_id

        self._endpoint = OPENROUTER_BASE_URL.rstrip("/") + "/chat/completions"
        self._api_key = os.getenv(env_key, "")

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not self._api_key:
            raise RuntimeError(
                f"{self.name}: OPENROUTER_API_KEY が設定されていません。"
            )

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
        }

        # TARGET_TOKENS があり、かつ未指定なら max_tokens に反映
        if self.TARGET_TOKENS is not None and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = int(self.TARGET_TOKENS)

        payload.update(kwargs)

        try:
            resp = requests.post(
                self._endpoint,
                headers=headers,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.exception("%s: OpenRouter call failed", self.name)
            raise RuntimeError(f"{self.name}: OpenRouter call failed: {e}")

        return split_text_and_usage_from_dict(data)
