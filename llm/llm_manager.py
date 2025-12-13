from __future__ import annotations

from typing import Any, Dict, List, Tuple
import os

import streamlit as st  # ← 追加（secrets / session_state を見るため）

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


def _flag_enabled(key: str, default: bool = False) -> bool:
    """
    環境変数 or streamlit.secrets で Feature Flag を読む。
    - "1"/"true"/"yes"/"on" を True 扱い
    """
    v = os.getenv(key)
    if v is None:
        try:
            if isinstance(st.secrets, dict):
                v = st.secrets.get(key)  # type: ignore[assignment]
        except Exception:
            v = None
    if v is None:
        return default
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "on")


class LLMManager:
    """
    互換レイヤ。

    - 旧来の LLMManager API を維持
    - 実体は llm2.llm_ai.LLMAI に委譲
    """

    _POOL: Dict[str, "LLMManager"] = {}

    @classmethod
    def get_or_create(cls, persona_id: str = "default") -> "LLMManager":
        if persona_id in cls._POOL:
            return cls._POOL[persona_id]

        mgr = cls(persona_id=persona_id)
        cls._POOL[persona_id] = mgr
        return mgr

    def __init__(self, persona_id: str = "default") -> None:
        self.persona_id = persona_id

        # 新中枢
        self._llm_ai = LLMAI(persona_id=persona_id)

        # --- 標準モデル登録 ---
        register_gpt51(self._llm_ai)
        register_gpt4o(self._llm_ai)
        register_grok(self._llm_ai)
        register_gemini(self._llm_ai)
        register_llama_unc(self._llm_ai)

        # =========================================================
        # Hermes は「明示的に有効化された時だけ」登録する
        # （ここが“勝手に復活”の元凶だった）
        # =========================================================
        enable_hermes_old = _flag_enabled("LYRA_ENABLE_HERMES_OLD", default=False)
        enable_hermes_new = _flag_enabled("LYRA_ENABLE_HERMES_NEW", default=False)

        if enable_hermes_old:
            register_hermes_old(self._llm_ai)
        if enable_hermes_new:
            register_hermes_new(self._llm_ai)

    # ===========================================================
    # 互換API
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
            usage = result[1] if len(result) >= 2 and saysinstance(result[1], dict) else {}
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

    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        return self._llm_ai.get_model_props()

    def get_models_sorted(self) -> Dict[str, Dict[str, Any]]:
        return self._llm_ai.get_models_sorted()

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        return self._llm_ai.get_available_models()

    def set_enabled_models(self, enabled: Dict[str, bool]) -> None:
        self._llm_ai.set_enabled_models(enabled)
