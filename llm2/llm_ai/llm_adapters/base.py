# llm2/llm_ai/llm_adapters/base.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class BaseLLMAdapter:
    """
    各ベンダーごとの LLM 呼び出しをカプセル化する基底クラス。

    - name:  論理モデル名（"gpt51", "grok", "gemini", "hermes", "llama_unc" など）
    - call:  (messages, **kwargs) -> (text, usage_dict or None)
    """

    name: str = ""

    # モデルごとの「推奨トークン長」（未指定なら Adapter 実装に任せる）
    TARGET_TOKENS: Optional[int] = None

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        raise NotImplementedError
