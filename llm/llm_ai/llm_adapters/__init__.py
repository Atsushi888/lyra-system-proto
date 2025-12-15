# llm/llm_ai/llm_adapters/__init__.py
"""
llm.llm_ai.llm_adapters

各 LLM ベンダーごとの Adapter 実装群。

- BaseLLMAdapter を共通インターフェースとする
- 個別 Adapter はこの配下に分割して配置
"""

from llm.llm_ai.llm_adapters.base import BaseLLMAdapter

__all__ = [
    "BaseLLMAdapter",
]
