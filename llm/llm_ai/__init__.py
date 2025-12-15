# llm/llm_ai/__init__.py
"""
llm.llm_ai

次世代 LLM 管理レイヤ。

- LLMAI を唯一の公開エントリポイントとする
- Adapter / Register / Factory は下位モジュールに分離
"""

from llm.llm_ai.llm_ai import LLMAI

__all__ = [
    "LLMAI",
]
