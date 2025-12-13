# llm2/llm_ai/llm_adapters/grok.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import logging
import requests

from llm2.llm_ai.llm_adapters.base import BaseLLMAdapter
from llm2.llm_ai.llm_adapters.utils import split_text_and_usage_from_dict

logger = logging.getLogger(__name__)


class GrokAdapter(BaseLLMAdapter):
    """
    xAI Grok 用アダプタ。

    docs.x.ai の curl と同じ:
      POST https://api.x.ai/v1/chat/completions
      Authorization: Bearer $XAI_API_KEY
      model: grok-4 (など)
    """

    def __init__(
        self,
        *,
        name: str = "grok",
        model_id: str = "grok-4",
        # 公式は XAI_API_KEY。互換で GROK_API_KEY も読む
        env_key_primary: str = "XAI_API_KEY",
        env_key_fallback: str = "GROK_API_KEY",
    ) -> None:
        self.name = name
        self.model_id = model_id

        self._endpoint = "https://api.x.ai/v1/chat/completions"
        self._api_key = os.getenv(env_key_primary, "") or os.getenv(env_key_fallback, "")

        self.TARGET_TOKENS = 480

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not self._api_key:
            raise RuntimeError("XAI_API_KEY (or GROK_API_KEY) が設定されていません。")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
        }

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
            logger.exception("%s: Grok call failed", self.name)
            raise RuntimeError(f"{self.name}: Grok call failed: {e}")

        return split_text_and_usage_from_dict(data)
