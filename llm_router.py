# llm_router.py — GPT系 + Hermes フォールバック用

import os
import requests
from typing import Any, Dict, List, Tuple

from openai import OpenAI


# ====== 環境変数（初期値として保持するが "参考値" 扱い） ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# メイン側（GPT系）のモデル名
MAIN_MODEL = os.getenv("OPENAI_MAIN_MODEL", "gpt-4o")

# Hermes 側のモデル名（OpenRouter の公式 ID）
HERMES_MODEL = os.getenv(
    "OPENROUTER_HERMES_MODEL",
    "nousresearch/hermes-3-llama-3.1-70b",
)

# ★ ここはもう使わないのでコメントアウト or 削除してOK
# client_openai = OpenAI(api_key=OPENAI_API_KEY)


# ====== GPT系（メイン） ======
def _call_gpt(
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    # ★ 呼び出し時点での環境変数を見る（LyraEngine が後からセットした値も拾える）
    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が設定されていません。")

    # ★ 毎回、その時点のキーでクライアントを作る
    client_openai = OpenAI(api_key=api_key)

    resp = client_openai.chat.completions.create(
        model=MAIN_MODEL,
        messages=messages,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )

    text = resp.choices[0].message.content or ""

    usage = {}
    if getattr(resp, "usage", None) is not None:
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "total_tokens": getattr(resp.usage, "total_tokens", None),
        }

    return text, usage


# ====== Hermes（フォールバック） ======
def _call_hermes(
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> Tuple[str, Dict[str, Any]]:
    # ★ こちらも都度 getenv で見る
    api_key = os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY が設定されていません。")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.io",
        "X-Title": "Lyra-Engine-Floria",
    }
    payload = {
        "model": HERMES_MODEL,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }

    r = requests.post(url, headers=headers, json=payload, timeout=(10, 60))
    r.raise_for_status()
    data = r.json()

    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content", "") or ""
    usage = data.get("usage", {})

    return text, usage


# ====== 公開インターフェース：フォールバック付き呼び出し ======
def call_with_fallback(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 800,
) -> Tuple[str, Dict[str, Any]]:
    """
    1. まず GPT 系（OpenAI）を試す
    2. エラーになったら Hermes（OpenRouter）を試す
    3. 両方ダメなら "" を返し、meta["route"] = "error" にする
    """
    meta: Dict[str, Any] = {}

    # 1) GPT 系
    try:
        text, usage = _call_gpt(messages, temperature, max_tokens)
        meta["route"] = "gpt"
        meta["model_main"] = MAIN_MODEL
        meta["usage_main"] = usage
        return text, meta
    except Exception as e:
        meta["gpt_error"] = str(e)

    # 2) Hermes フォールバック
    try:
        text, usage = _call_hermes(messages, temperature, max_tokens)
        meta["route"] = "hermes"
        meta["model_fallback"] = HERMES_MODEL
        meta["usage_fallback"] = usage
        return text, meta
    except Exception as e2:
        meta["route"] = "error"
        meta["hermes_error"] = str(e2)
        return "", meta
