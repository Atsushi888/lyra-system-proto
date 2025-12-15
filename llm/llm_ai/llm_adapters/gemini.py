# llm2/llm_ai/llm_adapters/gemini.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import logging
import requests

from llm.llm_ai.llm_adapters.base import BaseLLMAdapter

logger = logging.getLogger(__name__)


class GeminiAdapter(BaseLLMAdapter):
    """
    Google Gemini 2.0 用アダプタ。

    - OpenAI 互換ではないため REST API を直接呼び出す
    - Flash 系は短文になりがちなので、やや長めの maxOutputTokens を標準にする
    """

    def __init__(
        self,
        *,
        name: str = "gemini",
        model_id: str = "gemini-2.0-flash-exp",
        env_key: str = "GEMINI_API_KEY",
    ) -> None:
        self.name = name
        self.model_id = model_id

        self._api_key = os.getenv(env_key, "")
        self._endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_id}:generateContent"
        )

        # Flash らしさは保ちつつ、短すぎない程度
        self.TARGET_TOKENS = 400

    def _to_gemini_contents(
        self,
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """
        OpenAI 互換 messages を Gemini contents 形式へ変換。
        """
        contents: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            text = m.get("content", "") or ""
            contents.append(
                {
                    "role": "user" if role != "assistant" else "model",
                    "parts": [{"text": text}],
                }
            )
        return contents

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY が設定されていません。")

        # OpenAI / Lyra 内部用パラメータは捨てる
        for k in ("mode", "judge_mode", "max_tokens", "max_completion_tokens"):
            kwargs.pop(k, None)

        params: Dict[str, Any] = {
            "contents": self._to_gemini_contents(messages),
        }

        # generationConfig 未指定なら TARGET_TOKENS を反映
        if self.TARGET_TOKENS is not None and "generationConfig" not in kwargs:
            kwargs["generationConfig"] = {
                "maxOutputTokens": int(self.TARGET_TOKENS),
            }

        params.update(kwargs)

        try:
            resp = requests.post(
                self._endpoint,
                params={"key": self._api_key},
                json=params,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.exception("%s: Gemini call failed", self.name)
            raise RuntimeError(f"{self.name}: Gemini call failed: {e}")

        text = ""
        try:
            cands = data.get("candidates") or []
            if cands:
                parts = cands[0].get("content", {}).get("parts", [])
                if parts:
                    text = parts[0].get("text", "") or ""
        except Exception:
            logger.exception("Gemini response parse error")

        return text, None
