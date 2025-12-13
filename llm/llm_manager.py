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

# ※ gpt52 を起こす場合：この register が存在する前提
#   まだ無いなら作ってね（register_gpt51 と同様の構造でOK）
try:
    from llm2.llm_ai.llm_registers.register_gpt52 import register_gpt52  # type: ignore
    _HAS_GPT52 = True
except Exception:
    register_gpt52 = None  # type: ignore
    _HAS_GPT52 = False


class LLMManager:
    """
    互換レイヤ。

    - 旧来の LLMManager API を維持
    - 実体は llm2.llm_ai.LLMAI に委譲

    モデルの「登録」を環境変数で制御する。
    → 登録されないモデルは UI に出ない / 呼び出せない（ゾンビ封印）
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

        # --- 登録制御（ここが肝） ---------------------------------
        # 例:
        #   LYRA_ENABLE_MODELS="gpt51,gpt52,grok,gemini"
        #   LYRA_DISABLE_MODELS="gpt4o,hermes,hermes_new,llama_unc"
        #
        # enable を指定しない場合のデフォルトは「安全側」：
        #   gpt51 / (あれば gpt52) / grok / gemini のみ
        enable_raw = os.getenv("LYRA_ENABLE_MODELS", "").strip()
        disable_raw = os.getenv("LYRA_DISABLE_MODELS", "").strip()

        enable_set = {
            s.strip().lower() for s in enable_raw.split(",") if s.strip()
        }
        disable_set = {
            s.strip().lower() for s in disable_raw.split(",") if s.strip()
        }

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

        # --- 標準モデル登録（必要なものだけ登録） --------------------
        if want("gpt51"):
            register_gpt51(self._llm_ai)

        # gpt52 を起こす（register が存在するときのみ）
        if want("gpt52") and _HAS_GPT52 and register_gpt52 is not None:
            register_gpt52(self._llm_ai)

        # gpt4o は「眠らせたい」ことが多いのでデフォルトでは登録しない
        if want("gpt4o"):
            register_gpt4o(self._llm_ai)

        if want("grok"):
            register_grok(self._llm_ai)

        if want("gemini"):
            register_gemini(self._llm_ai)

        # --- OpenRouter 系（Hermes / llama_unc）はデフォルト封印 ----
        # ここを登録しない限り UI にも出ない＝復活しない
        if want("hermes") or want("hermes_old"):
            register_hermes_old(self._llm_ai)

        if want("hermes_new"):
            register_hermes_new(self._llm_ai)

        if want("llama_unc"):
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

        # tuple (text, usage)
        if isinstance(result, tuple) and len(result) >= 1:
            text = str(result[0] or "")
            usage = result[1] if len(result) >= 2 and isinstance(result[1], dict) else {}
            return text, usage

        # dict
        if isinstance(result, dict):
            text = str(
                result.get("text")
                or result.get("content")
                or result.get("message")
                or ""
            )
            usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
            return text, usage

        # fallback
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
