# actors/llm_adapters/gpt51_ai.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os

from openai import OpenAI, BadRequestError  # type: ignore

from actors.llm_ai import LLMAI
from actors.llm_adapters.emotion_style_prompt import (
    inject_emotion_style_system_prompt,
)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GPT51_MODEL = os.getenv("GPT51_MODEL", "gpt-5.1")


class GPT51AI(LLMAI):
    """
    gpt-5.1 用 LLMAI サブクラス。
    旧 LLMRouter.call_gpt51 のロジックをほぼそのまま内包する。
    """

    def __init__(self, enabled: bool = True, max_tokens: Optional[int] = None) -> None:
        super().__init__(
            name="gpt51",
            family="gpt-5.1",
            modes=["all"],    # 全 judge_mode で参加
            enabled=enabled,
        )
        self._client: Optional[OpenAI] = (
            OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        )
        # ★ llm_ai.py 側から渡せるヒント（指定なしなら従来通り 800）
        self.max_tokens: Optional[int] = max_tokens

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

        # ===== パラメータ整理 =====
        temperature = float(kwargs.pop("temperature", 0.7))
        top_p = float(kwargs.pop("top_p", 1.0))

        # llm_ai.py / ModelsAI2 から渡される max_tokens ヒント
        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is None and self.max_tokens is not None:
            max_tokens = int(self.max_tokens)
        if max_tokens is None:
            max_tokens = 800  # 従来デフォルト

        # 既存の system_prompt（あれば）
        user_system_prompt = kwargs.pop("system_prompt", None)

        # ★ 新パラメータ: emotion_style（EmotionResult / dict / JudgeSignal など）
        emotion_style = kwargs.pop("emotion_style", None)

        payload = messages

        if emotion_style is not None:
            # 感情スタイル＋既存 system をマージして先頭に差し込む
            payload = inject_emotion_style_system_prompt(
                messages=messages,
                hint_source=emotion_style,
                extra_system=user_system_prompt,
            )
        else:
            # 旧来通り、system_prompt があれば素直に先頭に付ける
            if user_system_prompt:
                payload = [{"role": "system", "content": user_system_prompt}] + messages

        # ===== 呼び出し本体 =====
        try:
            resp = self._client.chat.completions.create(
                model=GPT51_MODEL,
                messages=payload,
                temperature=temperature,
                top_p=top_p,
                max_completion_tokens=int(max_tokens),
            )
        except BadRequestError as e:  # type: ignore
            raise RuntimeError(f"GPT51AI BadRequestError: {e}") from e

        raw = resp.choices[0].message.content
        if not raw:
            raise RuntimeError(f"GPT51AI: empty content: {resp}")

        usage = self._extract_usage(resp)
        return raw, usage
