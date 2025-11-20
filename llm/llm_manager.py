# llm/llm_manager.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os

# YAML は optional 扱い（無ければ RuntimeError を投げてフォールバック）
try:
    import yaml  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]


@dataclass
class LLMModelConfig:
    """
    1 モデル分の定義情報。
    """

    name: str
    router_fn: str
    label: str
    priority: float = 1.0
    vendor: str = "unknown"
    roles: List[str] = field(default_factory=lambda: ["main"])
    required_env: List[str] = field(default_factory=list)
    enabled: bool = True
    default_params: Dict[str, Any] = field(default_factory=dict)

    def is_available(self) -> bool:
        """
        enabled かつ required_env が全部そろっているなら True。
        """
        if not self.enabled:
            return False
        for key in self.required_env:
            if not os.getenv(key):
                return False
        return True


class LLMManager:
    """
    LLM 設定のハブ。

    - モデル定義（優先度 / vendor / 必要な env など）を一元管理
    - AnswerTalker / ModelsAI / JudgeAI2 はここから model_props をもらう
    - 設定元は:
        - YAML (config/llm_default.yaml)
        - もしくはコード内の register_xxx() でハードコード
    """

    def __init__(
        self,
        persona_id: str = "default",
        models: Optional[Dict[str, LLMModelConfig]] = None,
    ) -> None:
        self.persona_id = persona_id
        self.models: Dict[str, LLMModelConfig] = models or {}

    # ============================
    # YAML からロード
    # ============================
    @classmethod
    def from_yaml(cls, path: str, persona_id: str = "default") -> "LLMManager":
        """
        config/llm_default.yaml などからモデル一覧をロードする。

        YAML 形式:
            models:
              gpt4o:
                label: "GPT-4o"
                router_fn: "call_gpt4o"
                priority: 3.0
                vendor: "openai"
                roles: ["main", "refiner"]
                required_env: ["OPENAI_API_KEY"]
                enabled: true
                default_params:
                  temperature: 0.9
                  max_tokens: 900
        """
        if yaml is None:
            raise RuntimeError(
                "PyYAML がインストールされていないため、YAML からのロードができません。"
                " `pip install pyyaml` を実行してください。"
            )

        if not os.path.exists(path):
            raise FileNotFoundError(f"LLM 設定ファイルが見つかりません: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        raw_models = data.get("models") or {}
        if not isinstance(raw_models, dict):
            raise RuntimeError("YAML のフォーマットが不正です（models が dict ではありません）。")

        models: Dict[str, LLMModelConfig] = {}
        for name, raw in raw_models.items():
            if not isinstance(raw, dict):
                continue
            router_fn = raw.get("router_fn")
            if not router_fn:
                # router_fn が無いモデルはスキップ
                continue

            cfg = LLMModelConfig(
                name=name,
                router_fn=str(router_fn),
                label=str(raw.get("label", name)),
                priority=float(raw.get("priority", 1.0)),
                vendor=str(raw.get("vendor", "unknown")),
                roles=list(raw.get("roles", ["main"])),
                required_env=list(raw.get("required_env", [])),
                enabled=bool(raw.get("enabled", True)),
                default_params=raw.get("default_params") or {},
            )
            models[name] = cfg

        return cls(persona_id=persona_id, models=models)

    # ============================
    # 手動登録 API
    # ============================
    def register_model(
        self,
        config: LLMModelConfig,
        enabled: Optional[bool] = None,
    ) -> None:
        """
        1 モデルを登録する共通ユーティリティ。
        """
        if enabled is not None:
            config.enabled = enabled
        self.models[config.name] = config

    def register_gpt4o(self, priority: float = 3.0, enabled: bool = True) -> None:
        self.register_model(
            LLMModelConfig(
                name="gpt4o",
                router_fn="call_gpt4o",
                label="GPT-4o",
                priority=priority,
                vendor="openai",
                roles=["main", "refiner"],
                required_env=["OPENAI_API_KEY"],
                enabled=enabled,
            )
        )

    def register_gpt51(self, priority: float = 4.0, enabled: bool = True) -> None:
        self.register_model(
            LLMModelConfig(
                name="gpt51",
                router_fn="call_gpt51",
                label="GPT-5.1",
                priority=priority,
                vendor="openai",
                roles=["main", "memory"],
                required_env=["OPENAI_API_KEY", "GPT51_MODEL"],
                enabled=enabled,
            )
        )

    def register_hermes(self, priority: float = 2.0, enabled: bool = True) -> None:
        self.register_model(
            LLMModelConfig(
                name="hermes",
                router_fn="call_hermes",
                label="Hermes 4",
                priority=priority,
                vendor="openrouter",
                roles=["main"],
                required_env=["OPENROUTER_API_KEY"],
                enabled=enabled,
            )
        )

    # ============================
    # 下流（ModelsAI / JudgeAI2）向け props 生成
    # ============================
    def get_model_props(
        self,
        role: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        下流でそのまま使える dict を返す。

        戻り値のイメージ:
            {
              "gpt4o": {
                "router_fn": "call_gpt4o",
                "label": "GPT-4o",
                "priority": 3.0,
                "vendor": "openai",
                "enabled": True,
                "required_env": [...],
                "default_params": {...},
              },
              ...
            }
        """
        result: Dict[str, Dict[str, Any]] = {}

        for name, cfg in self.models.items():
            if role and role not in cfg.roles:
                continue

            result[name] = {
                "router_fn": cfg.router_fn,
                "label": cfg.label,
                "priority": cfg.priority,
                "vendor": cfg.vendor,
                # env が足りているかどうかを反映した enabled
                "enabled": cfg.is_available(),
                "required_env": list(cfg.required_env),
                "default_params": dict(cfg.default_params),
                "roles": list(cfg.roles),
            }

        return result

    # ============================
    # UI 用ステータスサマリ
    # ============================
    def get_status_summary(self) -> Dict[str, Any]:
        """
        「どのモデルが使えるか」を UI に表示するためのメタ情報。
        """
        items: List[Dict[str, Any]] = []

        for name, cfg in self.models.items():
            missing = [k for k in cfg.required_env if not os.getenv(k)]
            items.append(
                {
                    "name": name,
                    "label": cfg.label,
                    "vendor": cfg.vendor,
                    "roles": list(cfg.roles),
                    "enabled": cfg.enabled,
                    "available": cfg.enabled and not missing,
                    "missing_env": missing,
                    "priority": cfg.priority,
                }
            )

        return {
            "persona_id": self.persona_id,
            "models": items,
        }
