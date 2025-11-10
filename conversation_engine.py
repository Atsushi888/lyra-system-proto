# conversation_engine.py

from typing import Any, Dict, List, Tuple

from deliberation.ai_response_collector import MultiAIResponseCollector
from components.multi_ai_response import PARTICIPATING_MODELS


class LLMConversation:
    ...

    def __init__(
        self,
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 800,
        style_hint: str = "",
    ) -> None:
        self.system_prompt = system_prompt
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.style_hint = style_hint.strip() if style_hint else ""

        self.default_style_hint = ( ... 省略 ... )

        # ★ ここで「審議に参加させるAI」を指定
        self.multi_ai = MultiAIResponseCollector(
            participants=list(PARTICIPATING_MODELS.keys()),
            primary="gpt4o",
        )

    ...

    def generate_reply(
        self,
        history: List[Dict[str, str]],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        会話履歴を受け取り、LLM応答テキストとメタ情報を返す。
        MultiAIResponseCollector を通じて複数AIの応答を集める。
        """
        messages = self.build_messages(history)

        # ★ 実際の呼び出しは全部 Collector に任せる
        text, meta = self.multi_ai.collect(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        meta = dict(meta)

        # Backstage 用情報を追記
        meta["prompt_messages"] = messages
        meta["prompt_preview"] = "\n\n".join(
            f"[{m['role']}] {m['content'][:300]}"
            for m in messages
        )

        return text, meta
