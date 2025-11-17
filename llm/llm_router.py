# llm/llm_router.py

from __future__ import annotations

import os
from typing import List, Dict, Any, Tuple

from openai import OpenAI


# ==== API クライアント & モデル名の設定 ===============================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Hermes 4 / GPT-5.1 のモデル名は環境変数で上書き可能にしておく。
# （OpenRouter 等で使う場合はここを実環境のモデル名に合わせて調整）
HERMES_MODEL = os.getenv("HERMES_MODEL", "hermes-4")
GPT51_MODEL = os.getenv("GPT51_MODEL", "gpt-5.1")


class LLMRouter:
    """
    GPT-4o / GPT-5.1 / Hermes などを呼び分ける中心クラス。

    - call_gpt4o  : OpenAI の GPT-4o
    - call_hermes : Hermes 4 系モデル（互換 API 前提）
    - call_gpt51  : GPT-5.1 系モデル（将来拡張を見据えた窓口）

    返り値はいずれも `(reply_text: str, usage: Dict[str, Any])` の 2 タプル。
    """

    def __init__(self, persona: Any = None):
        self.persona = persona

    # ============================
    # GPT-4o 呼び出し
    # ============================
    def call_gpt4o(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        GPT-4o を呼び出す最小ユニット。
        messages は Persona.build_messages() で組み立てた
        OpenAI 互換の messages 配列を想定。
        """
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

        reply_text = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }

        return reply_text, usage

    # ============================
    # Hermes 4 呼び出し
    # ============================
    def call_hermes(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Hermes 4 系モデルを呼び出す。

        - モデル名は環境変数 HERMES_MODEL で上書き可能。
        - OpenAI 互換 API（例：OpenRouter の OpenAI 互換エンドポイント）を想定。
        """
        resp = client.chat.completions.create(
            model=HERMES_MODEL,
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

        reply_text = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "total_tokens": getattr(resp.usage, "total_tokens", None),
        }

        return reply_text, usage

    # ============================
    # GPT-5.1 呼び出し
    # ============================
    def call_gpt51(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        GPT-5.1 系モデルを呼び出す窓口。

        - 実際のモデル名は環境変数 GPT51_MODEL で指定。
        - API 仕様は GPT-4o と同じく chat.completions を想定。
        """
        resp = client.chat.completions.create(
            model=GPT51_MODEL,
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

        reply_text = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "total_tokens": getattr(resp.usage, "total_tokens", None),
        }

        return reply_text, usage
