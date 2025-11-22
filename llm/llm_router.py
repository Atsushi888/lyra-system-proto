# llm/llm_router.py

from __future__ import annotations

import os
from typing import List, Dict, Any, Tuple

from openai import OpenAI, BadRequestError


# ===== OpenAI（GPT 系） =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
GPT4O_MODEL = os.getenv("OPENAI_MAIN_MODEL", "gpt-4o")
GPT51_MODEL = os.getenv("GPT51_MODEL", "gpt-5.1")

# ===== Hermes / OpenRouter =====
OPENROUTER_API_KEY_INITIAL = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
HERMES_MODEL = os.getenv("OPENROUTER_HERMES_MODEL", "nousresearch/hermes-4-70b")

# ===== Grok / xAI =====
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_BASE_URL = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-2-latest")

# ===== Gemini / Google =====
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai"
)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro-latest")


class LLMRouter:
    """
    GPT-4o / GPT-5.1 / Hermes / Grok / Gemini などを呼び分ける中心クラス。

    すべての call_xxx は
      -> Tuple[reply_text: str, usage: Dict[str, Any]]
    を返す。
    """

    def __init__(self, persona: Any = None):
        self.persona = persona

    # =====================================================================
    # utils
    # =====================================================================

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

    # =====================================================================
    # GPT-4o
    # =====================================================================
    def call_gpt4o(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 800,
        system_prompt: str | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        payload = messages
        if system_prompt:
            payload = [{"role": "system", "content": system_prompt}] + messages

        resp = client.chat.completions.create(
            model=GPT4O_MODEL,
            messages=payload,
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=int(max_tokens),
        )
        reply_text = resp.choices[0].message.content or ""
        return reply_text, self._extract_usage(resp)

    # =====================================================================
    # Hermes
    # =====================================================================
    def call_hermes(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 800,
        system_prompt: str | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        api_key = os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY_INITIAL
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY が設定されていません。")

        client_or = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

        payload = messages
        if system_prompt:
            payload = [{"role": "system", "content": system_prompt}] + messages

        try:
            resp = client_or.chat.completions.create(
                model=HERMES_MODEL,
                messages=payload,
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
            )
        except BadRequestError as e:
            raise RuntimeError(f"Hermes BadRequestError: {e}") from e

        reply_text = resp.choices[0].message.content or ""
        return reply_text, self._extract_usage(resp)

    # =====================================================================
    # GPT-5.1
    # =====================================================================
    def call_gpt51(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 800,
        system_prompt: str | None = None,
    ) -> Tuple[str, Dict[str, Any]]:

        payload = messages
        if system_prompt:
            payload = [{"role": "system", "content": system_prompt}] + messages

        resp = client.chat.completions.create(
            model=GPT51_MODEL,
            messages=payload,
            temperature=float(temperature),
            top_p=float(top_p),
            max_completion_tokens=int(max_tokens),
        )

        raw = resp.choices[0].message.content
        if not raw:
            raise RuntimeError(f"gpt51 returned empty content: {resp}")

        return raw, self._extract_usage(resp)

    # =====================================================================
    # Grok / xAI
    # =====================================================================
    def call_grok(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 800,
        system_prompt: str | None = None,
    ) -> Tuple[str, Dict[str, Any]]:

        if not GROK_API_KEY:
            raise RuntimeError("GROK_API_KEY が設定されていません。")

        client_grok = OpenAI(base_url=GROK_BASE_URL, api_key=GROK_API_KEY)

        payload = messages
        if system_prompt:
            payload = [{"role": "system", "content": system_prompt}] + messages

        try:
            resp = client_grok.chat.completions.create(
                model=GROK_MODEL,
                messages=payload,
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
            )
        except BadRequestError as e:
            raise RuntimeError(f"Grok BadRequestError: {e}") from e

        reply_text = resp.choices[0].message.content or ""
        return reply_text, self._extract_usage(resp)

    # =====================================================================
    # Gemini
    # =====================================================================
    def call_gemini(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 800,
        system_prompt: str | None = None,
    ) -> Tuple[str, Dict[str, Any]]:

        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY が設定されていません。")

        client_gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=GEMINI_API_KEY)

        payload = messages
        if system_prompt:
            payload = [{"role": "system", "content": system_prompt}] + messages

        try:
            resp = client_gemini.chat.completions.create(
                model=GEMINI_MODEL,
                messages=payload,
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=int(max_tokens),
            )
        except BadRequestError as e:
            raise RuntimeError(f"Gemini BadRequestError: {e}") from e

        reply_text = resp.choices[0].message.content or ""
        return reply_text, self._extract_usage(resp)
