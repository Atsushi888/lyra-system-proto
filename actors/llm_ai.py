# actors/llm_ai.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod


# ============================================================
# ベースクラス: LLMAI
# ============================================================

@dataclass
class LLMAI(ABC):
    """
    各 LLM ごとの共通インターフェース。

    - name:   論理名（"gpt51", "grok", "gemini", "hermes" など）
    - family: モデル系統名（"gpt-5.1", "grok-2", "gemini-2.0", "hermes-2" など）
    - modes:  参加モードのリスト
              例: ["normal"], ["erotic"], ["normal", "erotic"], ["all"]
    - enabled: 有効/無効フラグ
    - priority: 将来的に Judge 側での重みづけに利用予定
    """

    name: str
    family: str = ""
    modes: List[str] = field(default_factory=lambda: ["normal"])
    enabled: bool = True
    priority: float = 1.0

    # ---- 参加可否判定 -------------------------------------------------
    def should_answer(self, mode: str) -> bool:
        """
        現在の judge_mode (=mode) で、このモデルを実行すべきかどうか。

        - enabled=False → 常に不参加
        - modes に "all" が含まれていれば、mode に関係なく参加
        - それ以外は、mode（小文字）と modes の小文字を比較して一致したときのみ参加
        """
        if not self.enabled:
            return False

        if not mode:
            # mode 未指定なら enabled な限り参加
            return True

        mode_l = mode.lower()
        modes_l = [m.lower() for m in (self.modes or [])]

        if "all" in modes_l:
            return True

        return mode_l in modes_l

    # ---- 実際の呼び出し ------------------------------------------------
    @abstractmethod
    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        実際に LLM を叩いて (text, usage_dict or None) を返す。
        ここは各サブクラス（gpt51_ai / grok_ai / gemini_ai / hermes_xxx_ai）が実装する。
        """
        raise NotImplementedError


# ============================================================
# レジストリ: LLMAIRegistry
# ============================================================

class LLMAIRegistry:
    """
    LLMAI インスタンスを名前付きで管理する簡易レジストリ。

    ModelsAI2 からは:

        reg = LLMAIRegistry()
        reg.register(GPT51AI())
        reg.register(GrokAI())
        ...

        for name, ai in reg.all().items():
            ...

    のように利用する想定。
    """

    def __init__(self) -> None:
        self._models: Dict[str, LLMAI] = {}

    # ---- 登録系 --------------------------------------------------------
    def register(self, ai: LLMAI) -> None:
        """LLMAI インスタンスを name で登録する。"""
        self._models[ai.name] = ai

    def unregister(self, name: str) -> None:
        """名前を指定して登録を解除。なければ何もしない。"""
        self._models.pop(name, None)

    # ---- 取得系 --------------------------------------------------------
    def get(self, name: str) -> Optional[LLMAI]:
        """名前から LLMAI インスタンスを取得。存在しなければ None。"""
        return self._models.get(name)

    def all(self) -> Dict[str, LLMAI]:
        """全モデルの shallow copy を返す。"""
        return dict(self._models)

    # ---- メタ情報（デバッグ用）----------------------------------------
    def to_props(self) -> Dict[str, Dict[str, Any]]:
        """
        各モデルのメタ情報を dict で返す。
        Streamlit 側での表示・デバッグ用。
        """
        result: Dict[str, Dict[str, Any]] = {}
        for name, ai in self._models.items():
            result[name] = {
                "family": ai.family,
                "modes": list(ai.modes or []),
                "enabled": bool(ai.enabled),
                "priority": float(getattr(ai, "priority", 1.0)),
            }
        return result
