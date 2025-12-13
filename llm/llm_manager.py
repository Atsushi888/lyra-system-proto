# llm/llm_manager.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
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

# gpt52 は存在する/しないがまだ揺れてそうなので「あれば起こす」扱いにする
try:
    from llm2.llm_ai.llm_registers.register_gpt52 import register_gpt52  # type: ignore
    _HAS_GPT52 = True
except Exception:
    register_gpt52 = None  # type: ignore
    _HAS_GPT52 = False

try:
    import streamlit as st  # type: ignore
    _HAS_ST = True
except Exception:
    st = None  # type: ignore
    _HAS_ST = False


def _truthy(v: str | None) -> bool:
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on", "y")


def _has_key(env_key: str) -> bool:
    if os.getenv(env_key, ""):
        return True
    if _HAS_ST and isinstance(getattr(st, "secrets", None), dict) and st.secrets.get(env_key):
        return True
    return False


class LLMManager:
    """
    互換レイヤ。

    - 旧来の LLMManager API を維持
    - 実体は llm2.llm_ai.LLMAI に委譲

    方針：
    - 「眠らせる」= そもそも register しない（環境変数/フラグが無い限り復活しない）
    """

    _POOL: Dict[str, "LLMManager"] = {}

    # ===========================================================
    # singleton
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
    def __init__(self, persona_id: str = "default") -> None:
        self.persona_id = persona_id

        # 新中枢
        self._llm_ai = LLMAI(persona_id=persona_id)

        # -------------------------------------------------------
        # 登録ポリシー（重要）
        # -------------------------------------------------------
        # 1) OpenAI
        #   - gpt51: OPENAI_API_KEY があれば登録（基本主力）
        #   - gpt4o: デフォルトで「寝かせる」。必要なら ENABLE_GPT4O=1 で起こす
        #   - gpt52: 実装が存在し、かつ ENABLE_GPT52=1 なら起こす（なければ無視）
        #
        # 2) Grok/Gemini
        #   - それぞれ APIキーがある時だけ登録（=キー無しで復活しない）
        #
        # 3) Hermes/Llama(OpenRouter)
        #   - デフォルトで寝かせる。必要なら ENABLE_OPENROUTER=1 かつ OPENROUTER_API_KEY で起動
        # -------------------------------------------------------

        # --- OpenAI: gpt51 ---
        if _has_key("OPENAI_API_KEY"):
            register_gpt51(self._llm_ai)

            # gpt4o は基本オフ（明示フラグでオン）
            if _truthy(os.getenv("ENABLE_GPT4O")):
                register_gpt4o(self._llm_ai)

            # gpt52 は「実装が存在」かつ「明示フラグ」でオン
            if _HAS_GPT52 and _truthy(os.getenv("ENABLE_GPT52")):
                register_gpt52(self._llm_ai)  # type: ignore

        # --- xAI: Grok ---
        if _has_key("GROK_API_KEY"):
            register_grok(self._llm_ai)

        # --- Google: Gemini ---
        if _has_key("GEMINI_API_KEY"):
            register_gemini(self._llm_ai)

        # --- OpenRouter: Hermes/Llama ---
        openrouter_ok = _has_key("OPENROUTER_API_KEY") and _truthy(os.getenv("ENABLE_OPENROUTER"))
        if openrouter_ok:
            # old/new を分けたければフラグを追加で切れる
            # 例: ENABLE_HERMES_OLD / ENABLE_HERMES_NEW
            if _truthy(os.getenv("ENABLE_HERMES_OLD", "0")):
                register_hermes_old(self._llm_ai)
            if _truthy(os.getenv("ENABLE_HERMES_NEW", "0")):
                register_hermes_new(self._llm_ai)

            # llama_unc も同様
            if _truthy(os.getenv("ENABLE_LLAMA_UNC", "0")):
                register_llama_unc(self._llm_ai)

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
            usage = result[1] if len(result) >= 2 and isinstance(result[1], dict) else {}
            return text, usage

        if isinstance(result, dict):
            text = str(result.get("text") or result.get("content") or result.get("message") or "")
            usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
            return text, usage

        return str(result or ""), {}

    chat = chat_completion

    # ===========================================================
    # 情報取得系（ModelsAI2 用）
    # ===========================================================
    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        return self._llm_ai.get_model_props()

    def get_models_sorted(self) -> Dict[str, Dict[str, Any]]:
        return self._llm_ai.get_models_sorted()

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        return self._llm_ai.get_available_models()

    def set_enabled_models(self, enabled: Dict[str, bool]) -> None:
        self._llm_ai.set_enabled_models(enabled)
