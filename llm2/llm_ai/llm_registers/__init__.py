# llm2/llm_ai/llm_registers/__init__.py
"""
llm2.llm_ai.llm_registers

LLMAI に対して Adapter を登録するための register 群。

- 各 register_* 関数は LLMAI インスタンスを受け取り、
  対応する Adapter を生成して登録するだけの責務を持つ
- モデル追加・差し替えはこの配下を触るだけで完結する
"""

__all__ = []
