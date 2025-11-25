# actors/llm_ai.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import os

try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    st = None  # type: ignore
    _HAS_ST = False


class LLMAI:
    """
    各 LLM 用の共通ベースクラス。

    - name:   論理名（"gpt51", "grok", "gemini", "hermes" など）
    - family: ファミリ名（"gpt-5.1", "grok-2", "gemini-2.0", "hermes-2" など）
    - modes:  参加モード（["normal"], ["erotic"], ["all"] など）
    - enabled: 有効 / 無効
    - env_keys: この AI が使う API キーの環境変数名リスト
                （例: ["OPENAI_API_KEY"], ["GROK_API_KEY"] など）
    """

    def __init__(
        self,
        *,
        name: str,
        family: str,
        modes: List[str] | str = "all",
        enabled: bool = True,
        env_keys: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.family = family

        if isinstance(modes, str):
            self.modes = [modes]
        else:
            self.modes = list(modes)

        self.enabled = enabled
        self.env_keys = env_keys or []

    # --------------------------------------------------
    # 参加可否系
    # --------------------------------------------------
    def should_answer(self, mode: str) -> bool:
        """
        現在の judge_mode (=mode) で、この AI が回答に参加すべきか。
        """
        if not self.enabled:
            return False

        mode_key = (mode or "normal").lower()
        # "all" が含まれていれば常に参加
        lower_modes = [m.lower() for m in self.modes]
        if "all" in lower_modes:
            return True

        return mode_key in lower_modes

    def has_api_key(self) -> bool:
        """
        必要な API キーが環境変数 or streamlit.secrets に
        少なくとも 1 つは入っているかどうか。
        env_keys が空なら True 扱い（ノーチェック）。
        """
        if not self.env_keys:
            return True

        for key in self.env_keys:
            if os.getenv(key):
                return True
            if _HAS_ST and isinstance(st.secrets, dict) and st.secrets.get(key):
                return True

        return False

    # --------------------------------------------------
    # 実際の呼び出し（各サブクラスで実装）
    # --------------------------------------------------
    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError


class LLMAIRegistry:
    """
    name -> LLMAI インスタンス を管理するレジストリ。
    """

    def __init__(self) -> None:
        self._models: Dict[str, LLMAI] = {}

    def register(self, ai: LLMAI) -> None:
        self._models[ai.name] = ai

    def get(self, name: str) -> Optional[LLMAI]:
        return self._models.get(name)

    def all(self) -> Dict[str, LLMAI]:
        # 呼び出し側で安心して弄れるようにコピーを返す
        return dict(self._models)

    # --------------------------------------------------
    # デフォルト構成（gpt51 / grok / gemini / hermes*)
    # --------------------------------------------------
    @classmethod
    def create_default(cls) -> "LLMAIRegistry":
        """
        Lyra-System 標準の AI 構成でレジストリを組み立てる。
        """
        from actors.llm_adapters.gpt51_ai import GPT51AI
        from actors.llm_adapters.grok_ai import GrokAI
        from actors.llm_adapters.gemini_ai import GeminiAI
        from actors.llm_adapters.hermes_old_ai import HermesOldAI
        from actors.llm_adapters.hermes_new_ai import HermesNewAI

        reg = cls()
        reg.register(GPT51AI())
        reg.register(GrokAI())
        reg.register(GeminiAI())
        reg.register(HermesOldAI())
        reg.register(HermesNewAI())
        return reg
