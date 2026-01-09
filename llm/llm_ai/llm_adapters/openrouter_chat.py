# llm/llm_ai/llm_adapters/openrouter_chat.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import logging
import requests

from llm.llm_ai.llm_adapters.base import BaseLLMAdapter
from llm.llm_ai.llm_adapters.utils import split_text_and_usage_from_dict

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)


class OpenRouterChatAdapter(BaseLLMAdapter):
    """
    OpenRouter ChatCompletion 系（Hermes / Llama Uncensored など）の共通アダプタ。

    - requests で直接 OpenRouter API を叩く
    - max_tokens / TARGET_TOKENS の扱いを内部で統一
    - Persona由来/内部用の未知キーは送信前に除去（400事故防止）
    """

    # OpenRouter(OpenAI互換)で「まず安全に通る」トップレベルキー
    _ALLOW_PARAMS = {
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "stream",
        # 必要ならここに足す（例：response_format 等はモデルで死にやすいので慎重に）
    }

    # Lyra内部・OpenRouterへは送らないキー
    _DROP_PARAMS = {
        "mode",
        "judge_mode",
        "verbosity",
        "include_reasoning",
        "reasoning",
        "metadata",
        "response_format",
    }

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

    def _sanitize_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        k = dict(kwargs or {})

        # まず内部キーを確実に落とす
        for drop in self._DROP_PARAMS:
            k.pop(drop, None)

        # allowlist で絞る（未知キーを送らない）
        k = {key: val for key, val in k.items() if key in self._ALLOW_PARAMS and val is not None}

        return k

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

        # 送信前に sanitize
        safe_kwargs = self._sanitize_kwargs(kwargs)
        payload.update(safe_kwargs)

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
            # 返ってきた本文があるならログに残すとデバッグが一気に楽
            body = None
            try:
                body = resp.text  # type: ignore[name-defined]
            except Exception:
                pass

            logger.exception("%s: OpenRouter call failed payload_keys=%s body=%s",
                             self.name, sorted(payload.keys()), body)
            raise RuntimeError(f"{self.name}: OpenRouter call failed: {e}")

        return split_text_and_usage_from_dict(data)
