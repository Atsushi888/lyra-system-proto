# llm2/llm_ai/llm_adapters/openai_chat.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import logging

from openai import OpenAI as OpenAIClient

from llm2.llm_ai.llm_adapters.base import BaseLLMAdapter
from llm2.llm_ai.llm_adapters.utils import (
    split_text_and_usage_from_openai_completion,
    normalize_max_tokens,
)

logger = logging.getLogger(__name__)


class OpenAIChatAdapter(BaseLLMAdapter):
    """
    OpenAI ChatCompletion 系（GPT-4o / GPT-5.1 など）の共通アダプタ。

    - OpenAI SDK を直接使用
    - max_tokens / max_completion_tokens の差異を内部で吸収
    """

    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        env_key: str = "OPENAI_API_KEY",
    ) -> None:
        self.name = name
        self.model_id = model_id

        api_key = os.getenv(env_key, "")
        self._client: Optional[OpenAIClient] = (
            OpenAIClient(api_key=api_key) if api_key else None
        )

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if self._client is None:
            raise RuntimeError(f"{self.name}: OpenAI API キーが設定されていません。")

        # Lyra 内部用キーワードは削除
        for k in ("mode", "judge_mode"):
            kwargs.pop(k, None)

        # gpt-5.1 系だけは reasoning を弱めに
        if self.name == "gpt51" and "reasoning" not in kwargs:
            kwargs["reasoning"] = {"effort": "low"}

        # TARGET_TOKENS があり、かつ明示指定が無ければ適用
        if (
            self.TARGET_TOKENS is not None
            and "max_tokens" not in kwargs
            and "max_completion_tokens" not in kwargs
        ):
            kwargs["max_completion_tokens"] = int(self.TARGET_TOKENS)

        normalize_max_tokens(kwargs)

        last_exc: Optional[Exception] = None

        def _bump_max_tokens() -> None:
            inc = 160
            if "max_completion_tokens" in kwargs:
                cur = int(kwargs["max_completion_tokens"])
                kwargs["max_completion_tokens"] = min(cur + inc, 2048)
            elif "max_tokens" in kwargs:
                cur = int(kwargs["max_tokens"])
                kwargs["max_tokens"] = min(cur + inc, 2048)
            else:
                kwargs["max_completion_tokens"] = 512

        for attempt in range(3):
            try:
                completion = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    **kwargs,
                )

                text, usage = split_text_and_usage_from_openai_completion(completion)

                choices = getattr(completion, "choices", None) or []
                finish_reason = (
                    getattr(choices[0], "finish_reason", "") if choices else ""
                )

                if text.strip():
                    return text, usage

                if finish_reason == "length" and attempt < 2:
                    _bump_max_tokens()
                    continue

                return text, usage

            except Exception as e:
                last_exc = e
                logger.exception(
                    "%s: OpenAI call failed (attempt=%s)", self.name, attempt + 1
                )

        raise RuntimeError(
            f"{self.name}: OpenAI call failed after retry: {last_exc}"
        )
