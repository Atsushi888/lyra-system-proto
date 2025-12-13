# llm/llm_manager.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import os

# 新LLM中枢
from llm2.llm_ai.llm_ai import LLMAI

# register 群
from llm2.llm_ai.llm_registers.register_gpt51 import register_gpt51
from llm2.llm_ai.llm_registers.register_gpt4o import register_gpt4o
from llm2.llm_ai.llm_registers.register_grok import register_grok
from llm2.llm_ai.llm_registers.register_gemini import register_gemini
from llm2.llm_ai.llm_registers.register_hermes_old import register_hermes_old
from llm2.llm_ai.llm_registers.register_hermes_new import register_hermes_new
from llm2.llm_ai.llm_registers.register_llama_unc import register_llama_unc


def _env_flag(name: str, default: str = "0") -> bool:
    """
    "1/true/yes/on" を True として扱うENVフラグ。
    """
    v = os.getenv(name, default).strip().lower()
    return v in ("1", "true", "yes", "on", "y")


class LLMManager:
    """
    互換レイヤ。

    - 旧来の LLMManager API を維持
    - 実体は llm2.llm_ai.LLMAI に委譲
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
        # 登録ポリシー
        #   - “ENVがあるだけで勝手に復活” を防ぐため、
        #     OpenRouter系(Hermes/Llama)は明示フラグが必要。
        # -------------------------------------------------------
        has_openai = bool(os.getenv("OPENAI_API_KEY", ""))
        has_grok = bool(os.getenv("GROK_API_KEY", ""))
        has_gemini = bool(os.getenv("GEMINI_API_KEY", ""))
        has_openrouter = bool(os.getenv("OPENROUTER_API_KEY", ""))

        enable_openrouter_models = _env_flag("LYRA_ENABLE_OPENROUTER_MODELS", "0")

        # --- OpenAI系 ---
        # キーが無い環境でもUI側でモデル一覧を見せたいなら、ここは True 固定でもOK。
        # ただ「使えないモデルが混ざるのが嫌」なら has_openai でガード。
        if has_openai:
            register_gpt51(self._llm_ai)
            register_gpt4o(self._llm_ai)
        else:
            # “一覧だけ出す” をやりたいならここを有効化（ただし呼び出しは失敗する）
            # register_gpt51(self._llm_ai)
            # register_gpt4o(self._llm_ai)
            pass

        # --- xAI Grok ---
        if has_grok:
            register_grok(self._llm_ai)

        # --- Google Gemini ---
        if has_gemini:
            register_gemini(self._llm_ai)

        # --- OpenRouter系（Hermes / Llama） ---
        # 「ENVキーだけで勝手に復活」を止める：フラグ必須。
        if has_openrouter and enable_openrouter_models:
            register_hermes_old(self._llm_ai)
            register_hermes_new(self._llm_ai)
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
        """
        旧 AnswerTalker / ModelsAI2 用。

        戻り値形式は adapter に依存：
        - (text, usage) tuple
        - str
        """

        # -------------------------------------------------------
        # gpt51 が死ぬ件の “応急処置”
        #   ChatCompletions.create に reasoning= が飛ぶと即死する環境がある。
        #   ここで握りつぶして止血（根治は openai_chat adapter 側でフィルタが本筋）。
        # -------------------------------------------------------
        if model_name in ("gpt51", "gpt-5.1", "gpt5.1"):
            if "reasoning" in kwargs:
                kwargs.pop("reasoning", None)

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
        """
        ComposerAI / Refiner 用ラッパ。
        """
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

    # OpenAI 互換エイリアス
    chat = chat_completion

    # ===========================================================
    # 情報取得系（ModelsAI2 用）
    # ===========================================================
    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        """
        ModelsAI2 が参照するモデル一覧。
        """
        return self._llm_ai.get_model_props()

    def get_models_sorted(self) -> Dict[str, Dict[str, Any]]:
        return self._llm_ai.get_models_sorted()

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        return self._llm_ai.get_available_models()

    def set_enabled_models(self, enabled: Dict[str, bool]) -> None:
        self._llm_ai.set_enabled_models(enabled)
