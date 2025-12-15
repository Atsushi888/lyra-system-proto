from __future__ import annotations

from typing import Any

from llm.llm_ai.llm_adapters.openai_chat import OpenAIChatAdapter


def register_gpt52(llm_ai: Any) -> None:
    """
    GPT-5.2 Chat を LLMAI に登録する。

    重要:
    - ModelsAI2 側の安全フィルタのため、supported_parameters を extra に明示する
    - ここに無いキーは Persona から来ても自動で弾ける（ignored_params に回る）
    """
    adapter = OpenAIChatAdapter(
        name="gpt52",
        model_id="gpt-5.2-chat-latest",
    )

    # ★Persona → LLM に渡して良いトップレベルキーだけ許可
    # ここは Lyra の設計として運用しやすい粒度にしてOK（必要に応じて増やす）
    supported_parameters = [
        "temperature",
        "top_p",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "stop",
        # gpt52系の新パラメータ群（Personaから注入したいもの）
        "reasoning",
        "include_reasoning",
        "verbosity",
        "response_format",
        "metadata",
    ]

    llm_ai.register_adapter(
        adapter,
        extra={
            "supported_parameters": supported_parameters,
        },
    )
