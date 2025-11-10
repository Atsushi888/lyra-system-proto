# conversation_engine.py

from typing import Any, Dict, List, Tuple

from llm_router import call_with_fallback
from deliberation.multi_ai_response import PARTICIPATING_MODELS


class LLMConversation:
    ...

    def generate_reply(
        self,
        history: List[Dict[str, str]],
    ) -> Tuple[str, Dict[str, Any]]:
        messages = self.build_messages(history)

        # まず GPT-4o だけ呼ぶ（1回）
        text, meta = call_with_fallback(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        meta = dict(meta)

        meta["prompt_messages"] = messages
        meta["prompt_preview"] = "\n\n".join(
            f"[{m['role']}] {m['content'][:300]}"
            for m in messages
        )

        usage_main = meta.get("usage_main") or meta.get("usage") or {}

        models: Dict[str, Any] = {}

        # GPT-4o を models に詰める
        if "gpt4o" in PARTICIPATING_MODELS:
            models["gpt4o"] = {
                "reply": text,
                "usage": usage_main,
                "route": meta.get("route", "gpt"),
                "model_name": meta.get("model_main", "gpt-4o"),
            }

        # Hermes は今はダミー：GPT-4o の結果をコピー
        if "hermes" in PARTICIPATING_MODELS:
            models["hermes"] = {
                "reply": text,
                "usage": usage_main,
                "route": "dummy-hermes",
                "model_name": "Hermes (dummy)",
            }

        meta["models"] = models
        # もう top レベルの "gpt4o", "hermes" は作らない（JSONを細くする）

        return text, meta
