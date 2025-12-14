from __future__ import annotations

from typing import Any, Dict, List, Tuple
import os

from llm2.llm_ai import LLMAI

from llm2.llm_ai.llm_registers.register_gpt51 import register_gpt51
from llm2.llm_ai.llm_registers.register_gpt4o import register_gpt4o
from llm2.llm_ai.llm_registers.register_grok import register_grok
from llm2.llm_ai.llm_registers.register_gemini import register_gemini
from llm2.llm_ai.llm_registers.register_hermes_old import register_hermes_old
from llm2.llm_ai.llm_registers.register_hermes_new import register_hermes_new
from llm2.llm_ai.llm_registers.register_llama_unc import register_llama_unc

try:
    from llm2.llm_ai.llm_registers.register_gpt52 import register_gpt52  # type: ignore
    _HAS_GPT52 = True
except Exception:
    register_gpt52 = None  # type: ignore
    _HAS_GPT52 = False


class LLMManager:
    """
    互換レイヤ（デバッグ安全版）
    """

    _POOL: Dict[str, "LLMManager"] = {}

    @classmethod
    def get_or_create(cls, persona_id: str = "default") -> "LLMManager":
        if persona_id not in cls._POOL:
            cls._POOL[persona_id] = cls(persona_id=persona_id)
        return cls._POOL[persona_id]

    def __init__(self, persona_id: str = "default") -> None:
        self.persona_id = persona_id
        self._llm_ai = LLMAI(persona_id=persona_id)

        enable_raw = os.getenv("LYRA_ENABLE_MODELS", "").strip()
        disable_raw = os.getenv("LYRA_DISABLE_MODELS", "").strip()

        enable_set = {s.strip().lower() for s in enable_raw.split(",") if s.strip()}
        disable_set = {s.strip().lower() for s in disable_raw.split(",") if s.strip()}

        default_enable = {"gpt51", "grok", "gemini"}
        if _HAS_GPT52:
            default_enable.add("gpt52")

        def want(name: str) -> bool:
            key = name.lower()
            if key in disable_set:
                return False
            if enable_set:
                return key in enable_set
            return key in default_enable

        if want("gpt51"):
            register_gpt51(self._llm_ai)
        if want("gpt52") and _HAS_GPT52 and register_gpt52:
            register_gpt52(self._llm_ai)
        if want("gpt4o"):
            register_gpt4o(self._llm_ai)
        if want("grok"):
            register_grok(self._llm_ai)
        if want("gemini"):
            register_gemini(self._llm_ai)
        if want("hermes") or want("hermes_old"):
            register_hermes_old(self._llm_ai)
        if want("hermes_new"):
            register_hermes_new(self._llm_ai)
        if want("llama_unc"):
            register_llama_unc(self._llm_ai)

    # ---- 互換 API ----
    def call_model(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        return self._llm_ai.call(model_name=model_name, messages=messages, **kwargs)

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        result = self.call_model(model, messages, **kwargs)

        if isinstance(result, tuple):
            text = str(result[0] or "")
            usage = result[1] if len(result) > 1 and isinstance(result[1], dict) else {}
            return text, usage

        if isinstance(result, dict):
            return (
                str(result.get("text") or result.get("content") or ""),
                result.get("usage", {}),
            )

        return str(result or ""), {}

    chat = chat_completion

    # ---- 情報取得 ----
    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        return self._llm_ai.get_model_props()

    def get_models_sorted(self) -> Dict[str, Dict[str, Any]]:
        return self._llm_ai.get_models_sorted()

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        return self._llm_ai.get_available_models()

    def set_enabled_models(self, enabled: Dict[str, bool]) -> None:
        self._llm_ai.set_enabled_models(enabled)
