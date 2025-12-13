from __future__ import annotations

from typing import Any

from llm2.llm_ai.llm_adapters.gpt51 import GPT51Adapter


def register_gpt51(llm_ai: Any) -> None:
    """
    gpt51 を LLMAI に登録する。

    重要:
    - 古い OpenAI SDK / 呼び口では `reasoning=` が受け付けられず死ぬことがある。
      なので「デフォルト params」に reasoning を絶対に入れない。
    """
    adapter = GPT51Adapter()

    # ✅ reasoning は入れない（ここが肝）
    default_params = {
        # 必要ならここに共通パラメータだけ置く
        # "temperature": 0.7,
        # "max_tokens": 800,
    }

    llm_ai.register_adapter(
        adapter,
        vendor="openai",
        priority=1.0,
        enabled=True,
        params=default_params,
    )
