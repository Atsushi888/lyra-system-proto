# actors/llm_adapters/gemini_ai.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from actors.llm_ai import LLMAI
from llm.llm_adapter import GeminiAdapter
from actors.llm_adapters.emotion_style_prompt import (
    inject_emotion_style_system_prompt,
)


class GeminiAI(LLMAI):
    """
    Google Gemini 2.0 用 LLMAI。
    すべての judge_mode で参加。
    """

    def __init__(self, enabled: bool = True, max_tokens: Optional[int] = None) -> None:
        super().__init__(
            name="gemini",
            family="gemini-2.0",
            modes=["all"],
            enabled=enabled,
        )
        self._adapter = GeminiAdapter()
        self.max_tokens: Optional[int] = max_tokens

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        # Emotion 用
        emotion_style = kwargs.pop("emotion_style", None)
        user_system_prompt = kwargs.pop("system_prompt", None)

        payload = messages
        if emotion_style is not None:
            payload = inject_emotion_style_system_prompt(
                messages=messages,
                hint_source=emotion_style,
                extra_system=user_system_prompt,
            )
        elif user_system_prompt:
            payload = [{"role": "system", "content": user_system_prompt}] + messages

        # GeminiAdapter は generationConfig.maxOutputTokens を内部で設定している。
        # ここでは max_tokens をそのまま渡しておけば、adapter 側でよしなに扱えるようにする。
        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is None and self.max_tokens is not None:
            max_tokens = int(self.max_tokens)
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)

        return self._adapter.call(messages=payload, **kwargs)
