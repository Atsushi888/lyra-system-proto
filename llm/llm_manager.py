from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LLMModelConfig:
    """
    1つの LLM モデルに関する設定情報。

    - name:       論理名（"gpt4o", "gpt51", "hermes" など）
    - vendor:     ベンダー種別（"openai", "openrouter" など）
    - router_fn:  LLMRouter 内で呼び出すメソッド名（文字列）
    - priority:   優先度（大きいほど優先）
    - enabled:    有効フラグ
    - extra:      APIキー名や model_family などの追加情報
    """
    name: str
    vendor: str
    router_fn: str
    priority: float = 1.0
    enabled: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


class LLMManager:
    """
    LLM モデル群のメタ情報を管理するクラス。

    役割はあくまで「設定のレジストリ」であって、
    ここから直接 API 呼び出しは行わない。

    実際の呼び出しは ModelsAI 内の LLMRouter が、
    `router_fn` で指定されたメソッド名を使って行う想定。
    """

    # ★ ここを追加：persona_id ごとのシングルトン管理
    _POOL: Dict[str, "LLMManager"] = {}

    @classmethod
    def get_or_create(cls, persona_id: str = "default") -> "LLMManager":
        """
        persona_id ごとに LLMManager を 1 個だけ作って共有するヘルパ。

        - すでに作られていればそれを返す
        - なければ新規作成し、デフォルトモデルを登録してから保存する
        """
        # 既存があればそのまま返す
        if persona_id in cls._POOL:
            return cls._POOL[persona_id]

        # 新規作成
        manager = cls(persona_id=persona_id)

        # ★ ここは、llm_manager_factory に書いてあった初期登録ロジックをそのまま移植
        #    （llm_default.yaml の自動読込をあとで足したければ、ここに挿し込めばOK）
        manager.register_gpt4o(priority=3.0, enabled=True)
        manager.register_gpt51(priority=2.0, enabled=True)
        manager.register_hermes(priority=1.0, enabled=True)

        # プールに保存して、次回以降は同じインスタンスを返す
        cls._POOL[persona_id] = manager
        return manager
    
    def __init__(self, persona_id: str = "default", key: Optional[str] = None) -> None:
        """
        persona_id / key のどちらから呼ばれても動く後方互換仕様。
        旧コードで key="..." を渡している場合もここで吸収する。
        """
        # 旧来の key を優先しつつ、なければ persona_id を使う
        self.persona_id = key if key is not None else persona_id
    
        # name -> LLMModelConfig
        self._models: Dict[str, LLMModelConfig] = {}

    # ==============================
    # モデル登録系
    # ==============================
    def register_model(
        self,
        name: str,
        *,
        vendor: str,
        router_fn: str,
        priority: float = 1.0,
        enabled: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        汎用のモデル登録メソッド。
        """
        cfg = LLMModelConfig(
            name=name,
            vendor=vendor,
            router_fn=router_fn,
            priority=priority,
            enabled=enabled,
            extra=extra or {},
        )
        self._models[name] = cfg

    # --- 便利メソッド（標準3モデル） --------------------

    def register_gpt4o(self, *, priority: float = 3.0, enabled: bool = True) -> None:
        """
        gpt-4o 系のモデルを登録するためのヘルパ。
        実際の呼び出しは LLMRouter.call_gpt4o() を使う前提。
        """
        self.register_model(
            "gpt4o",
            vendor="openai",
            router_fn="call_gpt4o",
            priority=priority,
            enabled=enabled,
            extra={
                "env_key": "OPENAI_API_KEY",
                "model_family": "gpt-4o",
            },
        )

    def register_gpt51(self, *, priority: float = 2.0, enabled: bool = True) -> None:
        """
        gpt-5.1 系のモデルを登録するためのヘルパ。
        実際の呼び出しは LLMRouter.call_gpt51() を使う前提。
        """
        self.register_model(
            "gpt51",
            vendor="openai",
            router_fn="call_gpt51",
            priority=priority,
            enabled=enabled,
            extra={
                "env_key": "OPENAI_API_KEY",
                "model_family": "gpt-5.1",
            },
        )

    def register_hermes(self, *, priority: float = 1.0, enabled: bool = True) -> None:
        """
        Hermes 系（OpenRouter）のモデルを登録するためのヘルパ。
        実際の呼び出しは LLMRouter.call_hermes() を使う前提。
        """
        self.register_model(
            "hermes",
            vendor="openrouter",
            router_fn="call_hermes",
            priority=priority,
            enabled=enabled,
            extra={
                "env_key": "OPENROUTER_API_KEY",
                "model_family": "hermes",
            },
        )

    # ==============================
    # 取得系
    # ==============================
    def get_model_props(self) -> Dict[str, Dict[str, Any]]:
        """
        ModelsAI / View で使いやすいように dict 化した形で返す。

        戻り値の例:
        {
          "gpt4o": {
            "vendor": "openai",
            "router_fn": "call_gpt4o",
            "priority": 3.0,
            "enabled": True,
            "extra": {...},
          },
          ...
        }
        """
        result: Dict[str, Dict[str, Any]] = {}
        for name, cfg in self._models.items():
            result[name] = {
                "vendor": cfg.vendor,
                "router_fn": cfg.router_fn,
                "priority": cfg.priority,
                "enabled": cfg.enabled,
                "extra": dict(cfg.extra),
            }
        return result

    def get_models_sorted(self) -> Dict[str, Dict[str, Any]]:
        """
        priority の高い順にソート済みの dict を返すヘルパ。
        JudgeAI 側で優先度順に眺めたい時などに使える。
        """
        items = sorted(
            self._models.items(),
            key=lambda kv: kv[1].priority,
            reverse=True,
        )
        result: Dict[str, Dict[str, Any]] = {}
        for name, cfg in items:
            result[name] = {
                "vendor": cfg.vendor,
                "router_fn": cfg.router_fn,
                "priority": cfg.priority,
                "enabled": cfg.enabled,
                "extra": dict(cfg.extra),
            }
        return result

    # ==============================
    # YAML ロード（将来拡張用のダミー実装）
    # ==============================
    def load_default_config(self, path: Optional[str] = None) -> bool:
        """
        llm_default.yaml から設定を読み込むためのメソッド。

        いまは「ファイルがあれば読む / 無ければ False を返す」程度の実装にしておき、
        無くても動くようにしてある。

        YAML 形式の想定:
        models:
          gpt4o:
            vendor: openai
            router_fn: call_gpt4o
            priority: 3.0
            enabled: true
            extra:
              env_key: OPENAI_API_KEY
              model_family: gpt-4o
          ...
        """
        import os

        if path is None:
            path = "llm_default.yaml"

        if not os.path.exists(path):
            return False

        try:
            import yaml  # type: ignore[import]
        except Exception:
            # PyYAML が入っていなければ読み込みはスキップ
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return False

        models = data.get("models")
        if not isinstance(models, dict):
            return False

        for name, cfg in models.items():
            if not isinstance(cfg, dict):
                continue
            vendor = str(cfg.get("vendor", ""))
            router_fn = str(cfg.get("router_fn", ""))
            if not vendor or not router_fn:
                continue

            priority_raw = cfg.get("priority", 1.0)
            try:
                priority = float(priority_raw)
            except Exception:
                priority = 1.0

            enabled = bool(cfg.get("enabled", True))
            extra = cfg.get("extra") or {}
            if not isinstance(extra, dict):
                extra = {}

            self.register_model(
                name,
                vendor=vendor,
                router_fn=router_fn,
                priority=priority,
                enabled=enabled,
                extra=extra,
            )

        return True
