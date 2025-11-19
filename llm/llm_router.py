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


class LLMRouter:
    """
    GPT-4o / GPT-5.1 / Hermes などを呼び分ける中心クラス。

    すべての call_xxx は
      -> Tuple[reply_text: str, usage: Dict[str, Any]]
    を返す。
    """

    def __init__(self, persona: Any = None):
        self.persona = persona

    # ============================
    # GPT-4o 呼び出し（OpenAI）
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
            model=GPT4O_MODEL,
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

        reply_text = resp.choices[0].message.content or ""
        usage: Dict[str, Any] = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }

        return reply_text, usage

    # ============================
    # Hermes 4 呼び出し（OpenRouter）
    # ============================
    def call_hermes(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Hermes 4 系モデルを OpenRouter 経由で呼び出す。

        環境変数:
          - OPENROUTER_API_KEY
          - OPENROUTER_BASE_URL (任意, 既定: https://openrouter.ai/api/v1)
          - OPENROUTER_HERMES_MODEL (任意, 既定: nousresearch/hermes-4-70b)
        """
        api_key = os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY_INITIAL
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY が設定されていません。")

        client_or = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

        try:
            resp = client_or.chat.completions.create(
                model=HERMES_MODEL,
                messages=messages,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
        except BadRequestError as e:
            # ModelsAI 側で status:error に落ちるよう、例外として投げ直す
            raise RuntimeError(f"Hermes BadRequestError: {e}") from e

        reply_text = resp.choices[0].message.content or ""
        usage: Dict[str, Any] = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }

        return reply_text, usage

    # ============================
    # GPT-5.1 呼び出し（OpenAI）
    # ============================
    def call_gpt51(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        GPT-5.1 系モデルを呼び出す窓口。

        ※ このモデルは max_tokens ではなく max_completion_tokens を要求するため、
           パラメータ名に注意。
        """
        resp = client.chat.completions.create(
            model=GPT51_MODEL,
            messages=messages,
            temperature=float(temperature),
            max_completion_tokens=int(max_tokens),
        )

        # ★ ここを少しだけ強化する
        raw_content = resp.choices[0].message.content

        if not raw_content:
            # 中身が空なら「成功」とはみなさずエラーにしてしまう
            # → ModelsAI 側で status:error になり、llm_meta['models']['gpt51']['error']
            #    に resp 全体の情報が入るので原因を確認しやすくなる
            raise RuntimeError(f"gpt51 returned empty content: {resp}")

        reply_text = raw_content
        usage: Dict[str, Any] = {}
        if getattr(resp, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            }

        return reply_text, usage
