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


def _has_key(env_key: str) -> bool:
    """
    環境変数 or streamlit.secrets のどちらかに key が入っていれば True。
    """
    if not env_key:
        return True
    if os.getenv(env_key, ""):
        return True
    if _HAS_ST and isinstance(getattr(st, "secrets", None), dict) and st.secrets.get(env_key):
        return True
    return False


def _env_flag(name: str, default: str = "") -> str:
    """
    env or secrets からフラグ文字列を取る（secrets 優先）。
    """
    if _HAS_ST and isinstance(getattr(st, "secrets", None), dict) and st.secrets.get(name):
        return str(st.secrets.get(name) or "")
    return os.getenv(name, default)


def _flag_true(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "on", "enable", "enabled")


class LLMManager:
    """
    互換レイヤ。

    - 旧来の LLMManager API を維持
    - 実体は llm2.llm_ai.LLMAI に委譲

    方針：
    - “見せたくない/眠らせたいモデル” は **登録自体をしない**
      （enabled=False で残すのではなく、UIにも出さない）
    - 有効化は環境変数（or secrets）で制御する
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
        self._llm_ai = LLMAI(persona_id=persona_id)

        # =======================================================
        # まず “主要どころ” を登録（Hermes は条件付き）
        # =======================================================
        register_gpt51(self._llm_ai)
        register_gpt4o(self._llm_ai)
        register_grok(self._llm_ai)
        register_gemini(self._llm_ai)
        register_llama_unc(self._llm_ai)

        # =======================================================
        # Hermes は「復活」を防ぐため、明示フラグが無い限り登録しない
        #
        # 使うときだけ:
        #   LYRA_ENABLE_HERMES=1 かつ OPENROUTER_API_KEY が存在
        # =======================================================
        enable_hermes = _flag_true(_env_flag("LYRA_ENABLE_HERMES", "0"))
        if enable_hermes and _has_key("OPENROUTER_API_KEY"):
            register_hermes_old(self._llm_ai)
            register_hermes_new(self._llm_ai)

        # =======================================================
        # “キー無しモデルの参加” をここで抑止（登録はしてるが、無効化）
        # ※ UIには出るけど、呼ばれない。UIからも消したければ登録しない方針へ。
        # =======================================================
        self._apply_key_gates()

        # =======================================================
        # gpt51 の “reasoning” 事故を踏む環境向けの安全策
        # （openai SDK が古いと Completions.create が reasoning を受けず死ぬ）
        #
        # - 登録は維持しつつ、デフォルト params に reasoning が混ざっていたら除去
        # =======================================================
        self._strip_gpt51_reasoning_param()

    # ===========================================================
    # 内部: キー有無で enabled を落とす
    # ===========================================================
    def _apply_key_gates(self) -> None:
        props = self._llm_ai.get_model_props()
        enabled_map: Dict[str, bool] = {}

        for name, p in props.items():
            extra = p.get("extra") or {}
            env_key = str(extra.get("env_key") or "")
            if env_key and not _has_key(env_key):
                enabled_map[name] = False

        if enabled_map:
            self._llm_ai.set_enabled_models(enabled_map)

    # ===========================================================
    # 内部: gpt51 の reasoning をデフォルトから除去（SDK差異対策）
    # ===========================================================
    def _strip_gpt51_reasoning_param(self) -> None:
        try:
            # LLMAI の内部構造に触る（安全策。存在しない/構造違いなら握り潰す）
            models = getattr(self._llm_ai, "_models", None)
            if not isinstance(models, dict):
                return
            cfg = models.get("gpt51")
            if cfg is None:
                return
            params = getattr(cfg, "params", None)
            if isinstance(params, dict) and "reasoning" in params:
                params.pop("reasoning", None)
        except Exception:
            pass

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
        return self._llm_ai.get_model_props()

    def get_models_sorted(self) -> Dict[str, Dict[str, Any]]:
        return self._llm_ai.get_models_sorted()

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        return self._llm_ai.get_available_models()

    def set_enabled_models(self, enabled: Dict[str, bool]) -> None:
        self._llm_ai.set_enabled_models(enabled)
