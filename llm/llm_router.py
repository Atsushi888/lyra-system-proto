# llm/llm_router.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

# OpenAI / OpenRouter / xAI / Google など必要に応じて import
from openai import OpenAI
import requests
import json
import os


class LLMRouter:
    """
    各 LLM ベンダーごとの API 呼び出しを一元管理するルーター。
    モデル切替は LLMManager が行い、実際の呼び出しはここで処理する。
    """

    # ============================================================
    # ★ 共通ヘルパ：OpenAI ChatCompletion の content 正規化
    # ============================================================
    def _normalize_openai_content(self, message: Any) -> str:
        """
        OpenAI ChatCompletionMessage の content を安全に文字列として取り出す。

        - content が str: そのまま返す
        - content が list[dict(type="text", text=...)] の場合: text を結合
        - その他: 空文字扱い
        """
        content = getattr(message, "content", "")

        # 文字列ならそのまま
        if isinstance(content, str):
            return content or ""

        # list 形式（OpenAI v1 形式）
        if isinstance(content, list):
            parts: List[str] = []
            for chunk in content:
                if isinstance(chunk, dict):
                    if chunk.get("type") == "text":
                        parts.append(chunk.get("text") or "")
                else:
                    # pydantic モデル形式
                    t = getattr(chunk, "text", None)
                    if isinstance(t, str):
                        parts.append(t)
            return "".join(parts).strip()

        return ""

    # ============================================================
    # OpenAI: gpt-4o
    # ============================================================
    def call_gpt4o(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 800,
        **kwargs: Any,
    ):
        client = OpenAI()

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # 正規化
        msg = resp.choices[0].message
        text = self._normalize_openai_content(msg)

        usage = getattr(resp, "usage", None)
        usage_dict = {}
        if usage:
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }

        return text, usage_dict

    # ============================================================
    # OpenAI: gpt-5.1  ★エラー対策版
    # ============================================================
    def call_gpt51(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float = 0.8,
        max_tokens: int = 800,
        **kwargs: Any,
    ):
        """
        gpt-5.1 の呼び出し（v1 SDK）。
        content が空 or list の場合でも安全に処理し、
        空文字の場合は ValueError を投げる。
        """
        client = OpenAI()

        resp = client.chat.completions.create(
            model="gpt-5.1",
            messages=messages,
            temperature=float(kwargs.pop("temperature", temperature)),
            max_tokens=int(kwargs.pop("max_tokens", max_tokens)),
            **kwargs,
        )

        choice = resp.choices[0]
        msg = choice.message

        # ★ 正規化して内容取得
        text = self._normalize_openai_content(msg)

        # ★ content が完全に空のときは「不正」として例外を投げる
        if not text.strip():
            raise ValueError(f"gpt51 returned empty content: {resp!r}")

        usage = getattr(resp, "usage", None)
        usage_dict = {}
        if usage:
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }

        return text, usage_dict

    # ============================================================
    # Hermes（OpenRouter）
    # ============================================================
    def call_hermes(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 800,
        **kwargs: Any,
    ):
        """
        OpenRouter の Hermes など。
        """
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "nousresearch/hermes-3",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                          headers=headers, data=json.dumps(payload))
        data = r.json()

        choice = data["choices"][0]["message"]
        text = choice.get("content", "")

        usage = data.get("usage", {})
        return text, usage

    # ============================================================
    # Grok（xAI）
    # ============================================================
    def call_grok(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 800,
        **kwargs: Any,
    ):
        api_key = os.getenv("GROK_API_KEY", "")
        if not api_key:
            raise ValueError("GROK_API_KEY is not set.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": "grok-2",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        r = requests.post(
            "https://api.x.ai/v1/chat/completions", headers=headers, data=json.dumps(payload)
        )
        data = r.json()

        choice = data["choices"][0]["message"]
        text = choice.get("content", "")

        usage = data.get("usage", {})
        return text, usage

    # ============================================================
    # Gemini（Google）
    # ============================================================
    def call_gemini(
        self,
        *,
        messages: List[Dict[str, str]],
        temperature: float = 0.8,
        max_tokens: int = 800,
        **kwargs: Any,
    ):
        """
        Gemini API 呼び出し（仮実装、環境に合わせて書き換え可）
        """
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        # 必要に応じて Gemini API の正式仕様に合わせてね。
        # ここは最小限にしている。
        raise NotImplementedError("Gemini API is not yet implemented.")
