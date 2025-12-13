# llm2/llm_ai/llm_adapters/grok.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import logging

import requests

try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    st = None  # type: ignore
    _HAS_ST = False

from llm2.llm_ai.llm_adapters.base import BaseLLMAdapter
from llm2.llm_ai.llm_adapters.utils import split_text_and_usage_from_dict

logger = logging.getLogger(__name__)


class GrokAdapter(BaseLLMAdapter):
    """
    xAI Grok 用アダプタ。

    - OpenAI互換エンドポイントに POST
    - Streamlit Cloud 想定で st.secrets からも API key を読む
    - API key は __init__ で固定せず、call() のたびに取り直す（運用で差し替えやすい）
    """

    def __init__(
        self,
        *,
        name: str = "grok",
        model_id: str = "grok-2-latest",
        env_key: str = "GROK_API_KEY",
        endpoint: str = "https://api.x.ai/v1/chat/completions",
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.env_key = env_key
        self._endpoint = endpoint

        # GPT より少し長めでもよいくらい
        self.TARGET_TOKENS = 480

    # ==========================================================
    # key helper
    # ==========================================================
    def _get_api_key(self) -> str:
        v = os.getenv(self.env_key, "") or ""
        if v:
            return v
        if _HAS_ST and isinstance(st.secrets, dict):
            sv = st.secrets.get(self.env_key)
            if isinstance(sv, str) and sv.strip():
                return sv.strip()
        return ""

    # ==========================================================
    # parse helper (xAI format safety)
    # ==========================================================
    @staticmethod
    def _extract_text_usage_fallback(data: Dict[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        split_text_and_usage_from_dict が想定外フォーマットで落ちる/空になる時の保険。
        """
        text = ""

        # OpenAI互換: choices[0].message.content
        try:
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                c0 = choices[0] or {}
                msg = c0.get("message") if isinstance(c0, dict) else {}
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        text = content
        except Exception:
            pass

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        return text, usage

    # ==========================================================
    # call
    # ==========================================================
    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        api_key = self._get_api_key()
        if not api_key:
            raise RuntimeError(f"{self.env_key} が設定されていません（env or st.secrets）。")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
        }

        # TARGET_TOKENS があり、かつ未指定なら max_tokens に反映
        if self.TARGET_TOKENS is not None and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = int(self.TARGET_TOKENS)

        payload.update(kwargs)

        try:
            resp = requests.post(
                self._endpoint,
                headers=headers,
                json=payload,
                timeout=60,
            )

            # 失敗時に原因が見えるように、本文も少しログへ
            if resp.status_code >= 400:
                body_snip = (resp.text or "")[:800]
                logger.error(
                    "%s: Grok HTTP error status=%s body=%s",
                    self.name,
                    resp.status_code,
                    body_snip,
                )

            resp.raise_for_status()
            data = resp.json()

        except Exception as e:
            logger.exception("%s: Grok call failed", self.name)
            raise RuntimeError(f"{self.name}: Grok call failed: {e}")

        # まず既存ユーティリティで抽出
        try:
            text, usage = split_text_and_usage_from_dict(data)
        except Exception:
            text, usage = "", None

        # もし空なら fallback でもう一回拾う（「grokが喋らない」対策）
        if not str(text or "").strip():
            text2, usage2 = self._extract_text_usage_fallback(data if isinstance(data, dict) else {})
            if str(text2 or "").strip():
                text, usage = text2, usage2

        return str(text or ""), usage
