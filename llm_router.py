# llm_router.py

import os
from typing import Any, Dict, List, Tuple
from openai import OpenAI, BadRequestError  # ← 追加

OPENAI_API_KEY_INITIAL = os.getenv("OPENAI_API_KEY")
MAIN_MODEL = os.getenv("OPENAI_MAIN_MODEL", "gpt-4o")

# ★ OpenRouter / Hermes 用
OPENROUTER_API_KEY_INITIAL = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
HERMES_MODEL = os.getenv("OPENROUTER_HERMES_MODEL", "nousresearch/hermes-3")  # ← 実際に使うIDに変えてね


def _call_gpt(
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY_INITIAL
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が設定されていません。")

    client_openai = OpenAI(api_key=api_key)

    resp = client_openai.chat.completions.create(
        model=MAIN_MODEL,
        messages=messages,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )

    text = resp.choices[0].message.content or ""
    usage: Dict[str, Any] = {}
    if getattr(resp, "usage", None) is not None:
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "total_tokens": getattr(resp.usage, "total_tokens", None),
        }
    return text, usage

def _call_hermes(
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    api_key = os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY_INITIAL
    if not api_key:
        # キーが無いなら即ダミー返し
        return "[Hermes: OPENROUTER_API_KEY 未設定]", {
            "error": "OPENROUTER_API_KEY not set",
        }

    client_or = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )

    try:
        resp = client_or.chat.completions.create(
            model=HERMES_MODEL,
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
    except BadRequestError as e:
        # ★ ここで 400 を受け止めて、テキストとして返す
        return f"[Hermes BadRequestError: {e}]", {
            "error": str(e),
        }
    except Exception as e:
        # それ以外のエラーも一応
        return f"[Hermes Error: {e}]", {
            "error": str(e),
        }

    text = resp.choices[0].message.content or ""
    usage: Dict[str, Any] = {}
    if getattr(resp, "usage", None) is not None:
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "total_tokens": getattr(resp.usage, "total_tokens", None),
        }
    return text, usage

def call_with_fallback(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 800,
) -> Tuple[str, Dict[str, Any]]:
    """
    以前は GPT → Hermes フォールバックだったが、
    今は GPT-4o 単体のみをメインとして返す。
    """
    meta: Dict[str, Any] = {}
    try:
        text, usage = _call_gpt(messages, temperature, max_tokens)
        meta["route"] = "gpt"
        meta["model_main"] = MAIN_MODEL
        meta["usage_main"] = usage
        return text, meta
    except Exception as e:
        meta["route"] = "error"
        meta["gpt_error"] = str(e)
        return "", meta


# ★ Hermes 単体を呼ぶ公開関数
def call_hermes(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 800,
) -> Tuple[str, Dict[str, Any]]:
    text, usage = _call_hermes(messages, temperature, max_tokens)
    meta: Dict[str, Any] = {
        "route": "openrouter",
        "model_main": HERMES_MODEL,
        "usage_main": usage,
    }
    return text, meta
