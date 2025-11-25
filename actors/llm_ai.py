# actors/llm_ai.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ============================================================
# Base LLMAI
# ============================================================

class LLMAI(ABC):
    """
    個々の LLM をラップする共通インターフェイス。

    - name:   論理名（"gpt51" / "grok" / "gemini" / "hermes" など）
    - family: モデルファミリ（"gpt-5.1" / "grok-2" / "gemini-2.0" / "hermes-2" 等）
    - modes:  参加モード（["normal"], ["erotic"], ["all"] など）
    - enabled: 利用可否フラグ
    """

    name: str
    family: str
    modes: List[str]
    enabled: bool

    def __init__(
        self,
        name: str,
        family: str,
        modes: Optional[Iterable[str]] = None,
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.family = family
        self.modes = [m.lower() for m in (modes or ["all"])]
        self.enabled = enabled

    # -------------------------------------------
    # どのモードで参加するか
    # -------------------------------------------
    def should_answer(self, mode: str) -> bool:
        """
        現在の judge_mode（"normal" / "erotic" / …）に対して
        このモデルを呼ぶべきかどうかを判定。
        """
        if not self.enabled:
            return False

        if not self.modes:
            return True

        m = (mode or "normal").lower()

        # "all" を含んでいれば常に参加
        if "all" in self.modes:
            return True

        # 通常のモード一致判定
        return m in self.modes

    # -------------------------------------------
    # 実際の呼び出し
    # -------------------------------------------
    @abstractmethod
    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        LLM を呼び出し、(text, usage_dict or None) を返す。
        """
        raise NotImplementedError


# ============================================================
# Registry
# ============================================================

@dataclass
class LLMAIRegistry:
    """
    LLMAI サブクラスのレジストリ。

    - name（"gpt51" / "grok" / …）をキーに、LLMAI インスタンスを保持する。
    """

    models: Dict[str, LLMAI] = field(default_factory=dict)

    # -------------------------------------------
    # 登録まわり
    # -------------------------------------------
    def register(self, ai: LLMAI) -> None:
        self.models[ai.name] = ai

    def all(self) -> Dict[str, LLMAI]:
        """
        name -> LLMAI のディクショナリをコピーして返す。
        """
        return dict(self.models)

    def get(self, name: str) -> Optional[LLMAI]:
        return self.models.get(name)

    # -------------------------------------------
    # デフォルト構成の生成
    # -------------------------------------------
    @classmethod
    def create_default(cls) -> "LLMAIRegistry":
        """
        Lyra-System 既定の LLM 構成を生成して返す。

        gpt51 : 旧 router ベース
        grok  : GrokAdapter
        gemini: GeminiAdapter
        hermes: HermesOldAdapter（erotic 専用）
        hermes_new: HermesNewAdapter（デフォ無効）

        ※ import はここで遅延評価して、循環 import を避ける。
        """
        reg = cls()

        # ---- lazy imports ----
        from actors.llm_adapters.gpt51_ai import GPT51AI
        from actors.llm_adapters.grok_ai import GrokAI
        from actors.llm_adapters.gemini_ai import GeminiAI
        from actors.llm_adapters.hermes_old_ai import HermesOldAI
        from actors.llm_adapters.hermes_new_ai import HermesNewAI

        # 必須組
        reg.register(GPT51AI())
        reg.register(GrokAI())
        reg.register(GeminiAI())

        # Hermes 系
        reg.register(HermesOldAI())   # erotic 専用
        reg.register(HermesNewAI())   # デフォ無効だがレジストリ上は持っておく

        return reg
