from typing import Any, Dict, List, Tuple, Optional

class LLMManager:
    ...

    def call_model(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        *,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        return self._llm_ai.call(
            model_name=model_name,
            messages=messages,
            system_prompt=system_prompt,  # ★ そのまま中枢へ
            **kwargs,
        )

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        *,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        result = self.call_model(
            model,
            messages,
            system_prompt=system_prompt,
            **kwargs,
        )

        if isinstance(result, tuple) and len(result) >= 1:
            text = str(result[0] or "")
            usage = result[1] if len(result) >= 2 and isinstance(result[1], dict) else {}
            return text, usage

        if isinstance(result, dict):
            text = str(
                result.get("text")
                or result.get("content")
                or result.get("message")
                or ""
            )
            usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
            return text, usage

        return str(result or ""), {}

    chat = chat_completion
