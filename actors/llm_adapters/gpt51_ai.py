# actors/llm_adapters/gpt51_ai.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os

from openai import OpenAI, BadRequestError  # type: ignore

from actors.llm_ai import LLMAI
from actors.emotion_modes.emotion_style_prompt import EmotionStyle


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT51_MODEL = os.getenv("GPT51_MODEL", "gpt-5.1")


class GPT51AI(LLMAI):
    """
    gpt-5.1 用 LLMAI サブクラス。
    旧 LLMRouter.call_gpt51 のロジックをベースに、
    EmotionStyle による「感情 system プロンプト」をマージする。
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

    # ----------------------------------------
    # usage 抽出ヘルパ
    # ----------------------------------------
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

    # ----------------------------------------
    # 実コール
    # ----------------------------------------
    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if self._client is None:
            raise RuntimeError("GPT51AI: OPENAI_API_KEY が設定されていません。")

        # EmotionStyle（あれば system にマージ）
        emotion_style: Optional[EmotionStyle] = kwargs.pop("emotion_style", None)

        temperature = float(kwargs.pop("temperature", 0.7))
        top_p = float(kwargs.pop("top_p", 1.0))

        # max_tokens / max_completion_tokens 両対応
        max_completion_tokens = kwargs.pop("max_completion_tokens", None)
        max_tokens = kwargs.pop("max_tokens", None)

        if max_completion_tokens is None and max_tokens is not None:
            max_completion_tokens = int(max_tokens)

        if max_completion_tokens is None:
            # デフォルト：やや短め
            max_completion_tokens = 800

        system_prompt = kwargs.pop("system_prompt", None)

        # EmotionStyle の system プロンプトをマージ
        if emotion_style is not None:
            emo_sys = emotion_style.build_system_prompt()
            if system_prompt:
                system_prompt = system_prompt + "\n\n" + emo_sys
            else:
                system_prompt = emo_sys

        # system があれば先頭に付ける
        payload: List[Dict[str, str]] = list(messages)
        if system_prompt:
            payload = [{"role": "system", "content": system_prompt}] + payload

        try:
            resp = self._client.chat.completions.create(
                model=GPT51_MODEL,
                messages=payload,
                temperature=temperature,
                top_p=top_p,
                max_completion_tokens=int(max_completion_tokens),
            )
        except BadRequestError as e:  # type: ignore
            raise RuntimeError(f"GPT51AI BadRequestError: {e}") from e

        raw = resp.choices[0].message.content
        if not raw:
            raise RuntimeError(f"GPT51AI: empty content: {resp}")

        usage = self._extract_usage(resp)
        return raw, usage
