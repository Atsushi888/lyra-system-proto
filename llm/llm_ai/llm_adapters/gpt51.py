# llm2/llm_ai/llm_adapters/gpt51.py
from __future__ import annotations

from llm.llm_ai.llm_adapters.openai_chat import OpenAIChatAdapter


class GPT51Adapter(OpenAIChatAdapter):
    """
    gpt-5.1 用アダプタ。

    出力が長くなりがちなので、やや短めの max_completion_tokens をデフォルトにする。
    """

    def __init__(self) -> None:
        super().__init__(name="gpt51", model_id="gpt-5.1")
        # 感情豊かだがクドくなりすぎない程度
        self.TARGET_TOKENS = 220
