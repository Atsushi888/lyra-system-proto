from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import os

# 新LLM中枢
from llm2.llm_ai import LLMAI

# register 群
from llm2.llm_ai.llm_registers.register_gpt51 import register_gpt51
from llm2.llm_ai.llm_registers.register_gpt4o import register_gpt4o
from llm2.llm_ai.llm_registers.register_grok import register_grok
from llm2.llm_ai.llm_registers.register_gemini import register_gemini
from llm2.llm_ai.llm_registers.register_hermes_old import register_hermes_old
from llm2.llm_ai.llm_registers.register_hermes_new import register_hermes_new
from llm2.llm_ai.llm_registers.register_llama_unc import register_llama_unc

# gpt52 optional
try:
    from llm2.llm_ai.llm_registers.register_gpt52 import register_gpt52  # type: ignore
    _HAS_GPT52 = True
except Exception:
    register_gpt52 = None  # type: ignore
    _HAS_GPT52 = False


class LLMManager:
    """
    Lyra-System LLM 管理中枢（新版）

    - persona 単位シングルトン
    - 旧 LLMManager API 完全互換
    - 内部実体は llm2.llm_ai.LLMAI
    """

    _POOL: Dict[str, "LLMManager"] = {}

    # ===========================================================
    # factory / singleton
    # ===========================================================
    @classmethod
    def get_or_create(cls, persona_id: str = "default") -> "LLMManager":
        if persona_id in cls._POOL:
            return cls._POOL[persona_id]

        mgr = cls(persona_id=persona_id)
        cls._POOL[persona_id] = mgr
        return mgr

    # ===========================================================
    # init
    # ===========================================================
    def __init__(self, persona_id: Optional[str] = None) -> None:
        """
        ⚠️ 直接 new されても壊れないための防御設計
        """
        if persona_id is None:
            persona_id = "default"

        self.persona_id: str = persona_id
        self._initialized: bool = False

        # 新中枢
        self._llm_ai = LLMAI(persona_id=persona_id)

        # 初期化（多重実行防止）
        self._initialize_models()

    # ===========================================================
    # model register
    # ===========================================================
    def _initialize_models(self) -> None:
        if self._initialized:
            return

        enable_raw = os.getenv("LYRA_ENABLE_MODELS", "").strip()
        disable_raw = os.getenv("LYRA_DISABLE_MODELS", "").strip()

        enable_set = {s.strip().lower() for s in enable_raw.split(",") if s.strip()}
        disable_set = {s.strip().lower() for s in disable_raw.split(",") if s.strip()}

        default_enable = {"gpt51", "grok", "gemini"}
        if _HAS_GPT52:
            default_enable.add("gpt52")

        def want(name: str) -> bool:
            key = name.strip().lower()
            if key in disable_set:
                return False
            if enable_set:
                return key in enable_set
            return key in default_enable

        # ---- register ----
        if want("gpt51"):
            register_gpt51(self._llm_ai)

        if want("gpt52") and _HAS_GPT52 and register_gpt52 is not None:
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

        self._initialized = True

    # ===========================================================
    # legacy compatible API
    # ===========================================================
    def call_model(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        return self._llm_ai.call(
            model_name=model_name,
            messages=messages,
            **kwargs,
        )

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        result = self.call_model(model, messages, **kwargs)

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

    # ===========================================================
    # inspection / debug helpers
    # ===========================================================
    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        return self._llm_ai.get_model_props()

    def get_models_sorted(self) -> Dict[str, Dict[str, Any]]:
        return self._llm_ai.get_models_sorted()

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        return self._llm_ai.get_available_models()

    def set_enabled_models(self, enabled: Dict[str, bool]) -> None:
        self._llm_ai.set_enabled_models(enabled)

    def model_summary(self) -> Dict[str, Any]:
        """デバッグ・UI 表示用"""
        return {
            "persona_id": self.persona_id,
            "models": list(self.get_model_props().keys()),
            "initialized": self._initialized,
        }
