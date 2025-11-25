# actors/llm_ai.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class LLMAI:
    """
    各 LLM を共通インターフェースで扱うための基底クラス。

    - name:    論理名（"gpt51", "grok", "gemini", "hermes" など）
    - family:  モデルファミリ（"gpt-5.1", "grok-2", "gemini-2.0" など任意）
    - modes:   参加する judge_mode セット:
                 {"all"}      … 全モード参加
                 {"none"}     … どのモードでも参加しない
                 {"erotic"}   … erotic のときだけ参加
                 {"normal", "debate"} など任意組み合わせ
    - enabled: config 上の有効フラグ
    """

    def __init__(
        self,
        name: str,
        *,
        family: str = "",
        modes: Optional[List[str]] = None,
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.family = family or name
        self.enabled = bool(enabled)

        modes = modes or ["all"]
        self._modes = {m.lower() for m in modes}

    # ---- 参加可否 ----
    def should_answer(self, judge_mode: str) -> bool:
        """
        現在の judge_mode に対して、このモデルが回答候補に参加すべきかどうか。
        """
        if not self.enabled:
            return False

        if "none" in self._modes:
            return False

        if "all" in self._modes:
            return True

        mode = (judge_mode or "normal").lower()
        return mode in self._modes

    # ---- 実行本体（各サブクラスで実装） ----
    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        raise NotImplementedError


@dataclass
class LLMAIConfig:
    """
    LLMAIManager 内部で使う軽いメタ情報。
    """
    name: str
    family: str = ""
    modes: List[str] = field(default_factory=lambda: ["all"])
    enabled: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


class LLMAIManager:
    """
    LLMAI サブクラスを一括管理するマネージャ。
    - register(model): モデル登録
    - get(name):       1件取得
    - all():           dict で全件取得
    """

    def __init__(self) -> None:
        self._models: Dict[str, LLMAI] = {}
        self._configs: Dict[str, LLMAIConfig] = {}

    # -----------------------------
    # 登録
    # -----------------------------
    def register(self, model: LLMAI, *, extra: Optional[Dict[str, Any]] = None) -> None:
        self._models[model.name] = model
        cfg = LLMAIConfig(
            name=model.name,
            family=getattr(model, "family", model.name),
            modes=list(getattr(model, "_modes", {"all"})),
            enabled=bool(getattr(model, "enabled", True)),
            extra=extra or {},
        )
        self._configs[model.name] = cfg

    # -----------------------------
    # 取得系
    # -----------------------------
    def get(self, name: str) -> Optional[LLMAI]:
        return self._models.get(name)

    def all_models(self) -> Dict[str, LLMAI]:
        return dict(self._models)

    def all_configs(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "family": cfg.family,
                "modes": list(cfg.modes),
                "enabled": cfg.enabled,
                "extra": dict(cfg.extra),
            }
            for name, cfg in self._configs.items()
        }


# ============================================================
# デフォルト LLMAIManager ファクトリ
# ============================================================

def create_default_llm_ai_manager() -> LLMAIManager:
    """
    デフォルトの LLM 構成を持つ LLMAIManager を生成して返す。

    - gpt51:   すべての judge_mode に参加
    - grok:    すべての judge_mode に参加
    - gemini:  すべての judge_mode に参加
    - hermes:  erotic モードのときのみ参加
    - hermes_new: デフォルトでは無効（テスト時に有効化想定）
    - gpt4o:   デフォルトでは無効（将来用）
    """
    from actors.llm_adapters.gpt51_ai import GPT51AI
    from actors.llm_adapters.grok_ai import GrokAI
    from actors.llm_adapters.gemini_ai import GeminiAI
    from actors.llm_adapters.hermes_old_ai import HermesOldAI
    from actors.llm_adapters.hermes_new_ai import HermesNewAI
    from actors.llm_adapters.gpt4o_ai import GPT4oAI

    mgr = LLMAIManager()

    # GPT-5.1 … 全モード参加
    mgr.register(GPT51AI())

    # Grok … 全モード参加
    mgr.register(GrokAI())

    # Gemini … 全モード参加
    mgr.register(GeminiAI())

    # Hermes 旧 … erotic のときだけ有効
    mgr.register(HermesOldAI())

    # Hermes 新 … デフォルトでは無効（テスト時にクラス側で enabled を切り替え）
    mgr.register(HermesNewAI())

    # GPT-4o … 将来用。現在は enabled=False
    mgr.register(GPT4oAI())

    return mgr
