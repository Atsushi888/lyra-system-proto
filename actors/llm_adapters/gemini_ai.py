# actors/llm_adapters/gemini_ai.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from actors.llm_ai import LLMAI
from actors.emotion_modes.emotion_style_prompt import EmotionStyle
from llm.llm_adapter import GeminiAdapter  # 既存 adapter を再利用


class GeminiAI(LLMAI):
    """
    Google Gemini 2.0 用 LLMAI。
    EmotionStyle が渡された場合は、system メッセージとして先頭に付与する。
    """

    def __init__(self) -> None:
        super().__init__(
            name="gemini",
            family="gemini-2.0",
            modes=["all"],
            enabled=True,
        )
        self._adapter = GeminiAdapter()

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        # EmotionStyle（あれば system に反映）
        emotion_style: Optional[EmotionStyle] = kwargs.pop("emotion_style", None)

        payload: List[Dict[str, str]] = list(messages)
        if emotion_style is not None:
            sys_msg = {
                "role": "system",
                "content": emotion_style.build_system_prompt(),
            }
            payload = [sys_msg] + payload

        return self._adapter.call(messages=payload, **kwargs)
