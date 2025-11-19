# llm/llm_manager.py

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

from llm.llm_router import LLMRouter


@dataclass
class LLMModelConfig:
    """
    単一 LLM の設定情報。

    - name        : 内部名（"gpt4o" など）  ※一意キー
    - router_fn   : LLMRouter 上のメソッド名（"call_gpt4o" など）
    - label       : UI 表示用ラベル
    - priority    : JudgeAI2 などで使う優先度（大きいほど優先）
    - enabled     : 利用するかどうか
    - vendor      : ベンダ名（"openai" / "openrouter" など任意）
    - required_env: 要求される環境変数名リスト（表示用）
    """

    name: str
    router_fn: str
    label: str

    priority: float = 1.0
    enabled: bool = True
    vendor: str = ""
    required_env: List[str] = field(default_factory=list)


class LLMManager:
    """
    利用可能な LLM 一覧と、その呼び出しロジックを一元管理するクラス。

    - モデル定義の登録（register_* 系）
    - 利用可能なモデル一覧の取得
    - 実際の LLM 呼び出し（call_model）
    """

    def __init__(self, router: Optional[LLMRouter] = None) -> None:
        self.router: LLMRouter = router or LLMRouter()
        self._models: Dict[str, LLMModelConfig] = {}

    # ============================
    # モデル登録まわり
    # ============================
    def register_model(self, config: LLMModelConfig) -> None:
        """
        任意のモデル定義を登録する汎用メソッド。
        """
        self._models[config.name] = config

    # --- 便利ヘルパ（gpt4o / gpt51 / hermes） ---

    def register_gpt4o(
        self,
        priority: float = 3.0,
        enabled: bool = True,
    ) -> None:
        """
        OpenAI GPT-4o を標準設定で登録するヘルパ。
        """
        cfg = LLMModelConfig(
            name="gpt4o",
            router_fn="call_gpt4o",
            label="GPT-4o",
            priority=float(priority),
            enabled=enabled,
            vendor="openai",
            required_env=["OPENAI_API_KEY"],
        )
        self.register_model(cfg)

    def register_gpt51(
        self,
        priority: float = 2.0,
        enabled: bool = True,
    ) -> None:
        """
        OpenAI GPT-5.1 を標準設定で登録するヘルパ。
        """
        cfg = LLMModelConfig(
            name="gpt51",
            router_fn="call_gpt51",
            label="GPT-5.1",
            priority=float(priority),
            enabled=enabled,
            vendor="openai",
            required_env=["OPENAI_API_KEY"],
        )
        self.register_model(cfg)

    def register_hermes(
        self,
        priority: float = 1.0,
        enabled: bool = True,
    ) -> None:
        """
        OpenRouter Hermes 4 を標準設定で登録するヘルパ。
        """
        cfg = LLMModelConfig(
            name="hermes",
            router_fn="call_hermes",
            label="Hermes 4",
            priority=float(priority),
            enabled=enabled,
            vendor="openrouter",
            required_env=["OPENROUTER_API_KEY"],
        )
        self.register_model(cfg)

    # ============================
    # 参照系
    # ============================
    def get_model_config(self, name: str) -> Optional[LLMModelConfig]:
        return self._models.get(name)

    def list_model_configs(self) -> List[LLMModelConfig]:
        return list(self._models.values())

    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        """
        AnswerTalker → ModelsAI / JudgeAI2 へ渡すための、
        シンプルな dict 形式のプロパティ一覧を返す。

        戻り値例:
        {
          "gpt4o": {
            "enabled": True,
            "priority": 3.0,
            "router_fn": "call_gpt4o",
            "label": "GPT-4o",
            ...
          },
          ...
        }
        """
        props: Dict[str, Dict[str, Any]] = {}
        for name, cfg in self._models.items():
            props[name] = {
                "enabled": cfg.enabled,
                "priority": cfg.priority,
                "router_fn": cfg.router_fn,
                "label": cfg.label,
                "vendor": cfg.vendor,
                "required_env": list(cfg.required_env),
            }
        return props

    # ============================
    # 実際の LLM 呼び出し
    # ============================
    def call_model(
        self,
        name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        指定されたモデル名に対応する LLM を呼び出し、
        (reply_text, usage) を返す。

        - LLMRouter 上の router_fn を叩く
        - LLMRouter 側のシグネチャに処理を委譲する
        """
        cfg = self.get_model_config(name)
        if cfg is None:
            raise ValueError(f"unknown model: {name}")

        if not cfg.enabled:
            raise RuntimeError(f"model '{name}' is disabled")

        fn = getattr(self.router, cfg.router_fn, None)
        if fn is None:
            raise RuntimeError(f"router has no method '{cfg.router_fn}'")

        # LLMRouter 側で max_tokens / max_completion_tokens の差異を吸収している想定
        raw = fn(  # type: ignore[misc]
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

        # 返り値は LLMRouter の仕様に合わせて normalize しない
        # （呼び出し側が (text, usage) を想定して処理する）
        return raw

    # ============================
    # デバッグ用メタ情報
    # ============================
    def to_dict(self) -> Dict[str, Any]:
        """
        UI 表示などで使うための、単純な dict 化。
        """
        return {name: asdict(cfg) for name, cfg in self._models.items()}
