# llm/llm_router.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import os
import logging

from openai import OpenAI as OpenAIClient

import requests

logger = logging.getLogger(__name__)


class LLMRouter:
    """
    各ベンダーごとの LLM 呼び出しをまとめるルーター。

    - OpenAI: gpt4o / gpt5.1 など
    - OpenRouter: Hermes など
    - xAI: Grok
    - Google: Gemini

    AnswerTalker / ModelsAI からは
        router.call_gpt4o(messages=..., **kwargs)
    のように呼び出す想定。
    """

    # -------------------------------------------------
    # 初期化
    # -------------------------------------------------
    def __init__(self) -> None:
        # OpenAI
        openai_key = os.getenv("OPENAI_API_KEY", "")
        self._openai: Optional[OpenAIClient] = (
            OpenAIClient(api_key=openai_key) if openai_key else None
        )

        # OpenRouter
        self._openrouter_endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self._openrouter_key = os.getenv("OPENROUTER_API_KEY", "")

        # xAI (Grok)
        self._grok_endpoint = "https://api.x.ai/v1/chat/completions"
        self._grok_key = os.getenv("GROK_API_KEY", "")

        # Google (Gemini 2.0)
        self._gemini_endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.0-flash-exp:generateContent"
        )
        self._gemini_key = os.getenv("GEMINI_API_KEY", "")

    # -------------------------------------------------
    # 共通ユーティリティ
    # -------------------------------------------------
    @staticmethod
    def _to_openai_messages(
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        既に OpenAI 互換の {"role": "...", "content": "..."} 形式なら
        そのまま返すヘルパ。今後フォーマット変換が必要になったらここで行う。
        """
        return messages

    @staticmethod
    def _split_text_and_usage(
        completion: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        OpenAI / OpenRouter などの ChatCompletion から
        text と usage を取り出して返すヘルパ。
        """
        text = ""
        usage_dict: Optional[Dict[str, Any]] = None

        if hasattr(completion, "choices"):
            choices = getattr(completion, "choices") or []
            if choices:
                msg = getattr(choices[0], "message", None)
                if msg is not None:
                    text = getattr(msg, "content", "") or ""
        elif isinstance(completion, dict):
            # OpenRouter / 生 dict の場合
            choices = completion.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                text = msg.get("content", "") or ""

        # usage
        usage = getattr(completion, "usage", None)
        if usage is not None:
            try:
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }
            except Exception:
                usage_dict = None
        elif isinstance(completion, dict) and "usage" in completion:
            usage_dict = completion.get("usage")

        return text, usage_dict

    @staticmethod
    def _normalize_max_tokens(kwargs: Dict[str, Any]) -> None:
        """
        OpenAI の新 API では max_tokens ではなく max_completion_tokens を使う。
        - 呼び出し側が max_tokens を指定してきた場合：
            * max_completion_tokens が未指定ならコピー
            * その後 max_tokens キーは削除
        こうしておけば、gpt-5.1 のように max_tokens を拒否するモデルでも安全。
        """
        # 既に明示されていたらそちらを優先
        if "max_completion_tokens" in kwargs:
            kwargs.pop("max_tokens", None)
            return

        max_tokens = kwargs.pop("max_tokens", None)
        if max_tokens is not None:
            kwargs["max_completion_tokens"] = max_tokens

    # -------------------------------------------------
    # OpenAI 系
    # -------------------------------------------------
    def _call_openai_chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if self._openai is None:
            raise RuntimeError("OPENAI_API_KEY が設定されていません。")

        # max_tokens → max_completion_tokens 変換
        self._normalize_max_tokens(kwargs)

        oa_messages = self._to_openai_messages(messages)

        completion = self._openai.chat.completions.create(
            model=model,
            messages=oa_messages,
            **kwargs,
        )
        return self._split_text_and_usage(completion)

    def call_gpt4o(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        gpt-4o 呼び出し。
        既存コードから max_tokens を渡されても _normalize_max_tokens が面倒を見てくれる。
        """
        return self._call_openai_chat("gpt-4o-mini", messages, **kwargs)

    def call_gpt51(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        gpt-5.1 呼び出し。
        このモデルは `max_tokens` を受け付けず、`max_completion_tokens` を要求するため、
        _call_openai_chat 内で変換している。
        """
        return self._call_openai_chat("gpt-5.1", messages, **kwargs)

    # -------------------------------------------------
    # OpenRouter（Hermes）
    # -------------------------------------------------
    def call_hermes(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not self._openrouter_key:
            raise RuntimeError("OPENROUTER_API_KEY が設定されていません。")

        headers = {
            "Authorization": f"Bearer {self._openrouter_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": "nousresearch/hermes-3-llama-3.1-70b",
            "messages": messages,
        }
        payload.update(kwargs)

        resp = requests.post(
            self._openrouter_endpoint,
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return self._split_text_and_usage(data)

    # -------------------------------------------------
    # xAI（Grok）
    # -------------------------------------------------
    def call_grok(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not self._grok_key:
            raise RuntimeError("GROK_API_KEY が設定されていません。")

        headers = {
            "Authorization": f"Bearer {self._grok_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": "grok-2-latest",
            "messages": messages,
        }
        payload.update(kwargs)

        resp = requests.post(
            self._grok_endpoint,
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return self._split_text_and_usage(data)

    # -------------------------------------------------
    # Google（Gemini 2.0）
    # -------------------------------------------------
    def call_gemini(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not self._gemini_key:
            raise RuntimeError("GEMINI_API_KEY が設定されていません。")

        # 非常にラフだが、role/content から Gemini 用の contents に変換
        contents: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            text = m.get("content", "")
            contents.append(
                {
                    "role": "user" if role != "assistant" else "model",
                    "parts": [{"text": text}],
                }
            )

        params: Dict[str, Any] = {
            "contents": contents,
        }
        params.update(kwargs)

        resp = requests.post(
            self._gemini_endpoint,
            params={"key": self._gemini_key},
            json=params,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        # text 部だけ抜き出し（usage 相当は今は無し）
        text = ""
        try:
            cands = data.get("candidates") or []
            if cands:
                parts = (
                    cands[0]
                    .get("content", {})
                    .get("parts", [])
                )
                if parts:
                    text = parts[0].get("text", "") or ""
        except Exception:
            logger.exception("Gemini response parse error")

        return text, None
