# actors/llm_adapters/gpt51_ai.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os

from openai import OpenAI, BadRequestError  # type: ignore

from actors.llm_ai import LLMAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT51_MODEL = os.getenv("GPT51_MODEL", "gpt-5.1")


class GPT51AI(LLMAI):
    """
    gpt-5.1 用 LLMAI サブクラス。
    旧 LLMRouter.call_gpt51 のロジックをほぼそのまま内包する。
    """

    def __init__(self) -> None:
        super().__init__(
            name="gpt51",
            family="gpt-5.1",
            modes=["all"],    # 全 judge_mode で参加
            enabled=True,
        )
        self._client: Optional[OpenAI] = (
            OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        )

    @staticmethod
    def _extract_usage(resp: Any) -> Dict[str, Any]:
        usage: Dict[str, Any] = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }
        return usage

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if self._client is None:
            raise RuntimeError("GPT51AI: OPENAI_API_KEY が設定されていません。")

        temperature = float(kwargs.pop("temperature", 0.7))
        top_p = float(kwargs.pop("top_p", 1.0))
        max_tokens = int(kwargs.pop("max_tokens", 800))
        system_prompt = kwargs.pop("system_prompt", None)

        payload = messages
        if system_prompt:
            payload = [{"role": "system", "content": system_prompt}] + messages

        try:
            resp = self._client.chat.completions.create(
                model=GPT51_MODEL,
                messages=payload,
                temperature=temperature,
                top_p=top_p,
                max_completion_tokens=max_tokens,
            )
        except BadRequestError as e:  # type: ignore
            raise RuntimeError(f"GPT51AI BadRequestError: {e}") from e

        raw = resp.choices[0].message.content
        if not raw:
            raise RuntimeError(f"GPT51AI: empty content: {resp}")

        usage = self._extract_usage(resp)
        return raw, usage
