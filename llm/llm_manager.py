from __future__ import annotations

from typing import Any, Dict, List, Tuple
import os

try:
    import streamlit as st  # type: ignore
    _HAS_ST = True
except Exception:
    st = None  # type: ignore
    _HAS_ST = False

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


def _env_has(key: str) -> bool:
    if os.getenv(key):
        return True
    if _HAS_ST and isinstance(getattr(st, "secrets", None), dict) and st.secrets.get(key):
        return True
    return False


def _env_flag_true(key: str, default: bool = False) -> bool:
    """
    例:
      LYRASYSTEM_ENABLE_HERMES=1 / true / yes / on で True
      未設定なら default
    """
    v = os.getenv(key)
    if v is None:
        if _HAS_ST and isinstance(getattr(st, "secrets", None), dict):
            sv = st.secrets.get(key)
            if sv is None:
                return default
            v = str(sv)
        else:
            return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


class LLMManager:
    """
    互換レイヤ。

    - 旧来の LLMManager API を維持
    - 実体は llm2.llm_ai.LLMAI に委譲

    ★重要ポリシー（要望反映）
    - 「環境変数があるモデルだけ登録（=復活）する」
      -> APIキーが無ければ register 自体をしない（UIにも出ない）
    - 例外的に強制ON/OFFしたい時はフラグで制御
      LYRASYSTEM_ENABLE_OPENAI=1/0
      LYRASYSTEM_ENABLE_XAI=1/0
      LYRASYSTEM_ENABLE_GEMINI=1/0
      LYRASYSTEM_ENABLE_OPENROUTER=1/0
      LYRASYSTEM_ENABLE_HERMES=1/0
      LYRASYSTEM_ENABLE_LLAMA_UNC=1/0
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

        # ---- Provider 有効判定（キー or 明示フラグ）----
        enable_openai = _env_flag_true("LYRASYSTEM_ENABLE_OPENAI", default=True) and _env_has("OPENAI_API_KEY")
        enable_xai = _env_flag_true("LYRASYSTEM_ENABLE_XAI", default=True) and _env_has("GROK_API_KEY")
        enable_gemini = _env_flag_true("LYRASYSTEM_ENABLE_GEMINI", default=True) and _env_has("GEMINI_API_KEY")

        enable_openrouter = _env_flag_true("LYRASYSTEM_ENABLE_OPENROUTER", default=False) and _env_has("OPENROUTER_API_KEY")
        enable_hermes = _env_flag_true("LYRASYSTEM_ENABLE_HERMES", default=False)
        enable_llama_unc = _env_flag_true("LYRASYSTEM_ENABLE_LLAMA_UNC", default=False)

        # --- 標準モデル登録（キーがあるものだけ登録） ---
        if enable_openai:
            register_gpt51(self._llm_ai)
            register_gpt4o(self._llm_ai)

        if enable_xai:
            register_grok(self._llm_ai)

        if enable_gemini:
            register_gemini(self._llm_ai)

        # OpenRouter 系は「OPENROUTER_API_KEY がある」かつ「個別に許可」した場合のみ登録
        # ※ これで Hermes が勝手に復活しない（キーが無い/フラグOFFならレジストリに載らない）
        if enable_openrouter and enable_hermes:
            register_hermes_old(self._llm_ai)
            register_hermes_new(self._llm_ai)

        if enable_openrouter and enable_llama_unc:
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

        # ---- 安全弁：OpenAI ChatCompletions に渡すと落ちる引数を除去 ----
        # いま出ている: Completions/ChatCompletions.create() got unexpected keyword argument 'reasoning'
        # -> ここで剥がす（gpt51 だけでなく openai 系全体の事故防止）
        if model_name in ("gpt51", "gpt4o"):
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
