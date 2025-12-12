# llm2/llm_ai/llm_adapters/utils.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================
# OpenAI completion parser
# ============================================================
def split_text_and_usage_from_openai_completion(
    completion: Any,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    OpenAI ChatCompletion オブジェクトから text / usage を取り出す。

    gpt-5.1 のように message.content が list になるケースにも対応。
    """
    text = ""
    usage_dict: Optional[Dict[str, Any]] = None

    try:
        choices = getattr(completion, "choices", None) or []
        if choices:
            msg = getattr(choices[0], "message", None)
            if msg is not None:
                content = getattr(msg, "content", "") or ""

                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    parts: List[str] = []
                    for p in content:
                        t = getattr(p, "text", None)
                        if t is None and isinstance(p, dict):
                            t = p.get("text")
                        if t is None:
                            t = str(p)
                        parts.append(t)
                    text = "".join(parts)
                else:
                    text = str(content)
    except Exception:
        logger.exception("OpenAI completion parse error")
        text = ""

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

    return text, usage_dict


# ============================================================
# dict-based LLM parser (OpenRouter / Grok / others)
# ============================================================
def split_text_and_usage_from_dict(
    data: Dict[str, Any],
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    OpenRouter / Grok / その他 dict レスポンスから text / usage を取り出す。
    """
    text = ""
    usage_dict: Optional[Dict[str, Any]] = None

    try:
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            text = msg.get("content", "") or ""
    except Exception:
        logger.exception("LLM dict response parse error")
        text = ""

    if "usage" in data and isinstance(data.get("usage"), dict):
        usage_dict = data.get("usage")

    return text, usage_dict


# ============================================================
# token parameter normalizer (OpenAI)
# ============================================================
def normalize_max_tokens(kwargs: Dict[str, Any]) -> None:
    """
    OpenAI 新 API 向けに max_tokens / max_completion_tokens を正規化。

    - max_tokens が渡されてきた場合：
        * max_completion_tokens が無ければコピー
        * max_tokens は削除
    """
    if "max_completion_tokens" in kwargs:
        kwargs.pop("max_tokens", None)
        return

    max_tokens = kwargs.pop("max_tokens", None)
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens
