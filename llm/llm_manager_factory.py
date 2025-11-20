# llm/llm_manager_factory.py

from __future__ import annotations

from typing import Dict, Any
import os

import yaml  # pyyaml
from .llm_manager import LLMManager, LLMModelConfig

# persona_id ごとに 1 個だけ LLMManager を共有
_MANAGER_CACHE: Dict[str, LLMManager] = {}

# プロジェクト直下 or src 直下などに置く想定
DEFAULT_CONFIG_PATH = "llm_default.yaml"


def _load_from_yaml_if_exists(manager: LLMManager, path: str) -> bool:
    """
    llm_default.yaml があれば読み込んで manager に model を登録する。
    まともに読み込めたら True、なにかあってスキップしたら False。
    """
    if not os.path.exists(path):
        return False

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        # 壊れててもアプリごと死なないように False で返す
        return False

    models = data.get("models")
    if not isinstance(models, list):
        return False

    for item in models:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name", "")).strip()
        if not name:
            continue

        cfg = LLMModelConfig(
            name=name,
            router_fn=str(item.get("router_fn", "") or ""),
            label=str(item.get("label", "") or name),
            priority=float(item.get("priority", 1.0)),
            vendor=str(item.get("vendor", "") or ""),
            required_env=list(item.get("required_env") or []),
            enabled=bool(item.get("enabled", True)),
        )
        manager.register_model(cfg)

    return True


def get_llm_manager(persona_id: str = "default") -> LLMManager:
    """
    persona_id ごとに 1 個だけ LLMManager を生成してキャッシュする。

    優先順位:
      1) llm_default.yaml があればそれを読み込む
      2) 無ければ、コード内のデフォルト（gpt4o / gpt51 / hermes）を登録
    """
    key = persona_id or "default"

    # すでに作ってあればそれを返す
    if key in _MANAGER_CACHE:
        return _MANAGER_CACHE[key]

    manager = LLMManager(persona_id=key)

    # 1) YAML から読み込みを試みる
    loaded = _load_from_yaml_if_exists(manager, DEFAULT_CONFIG_PATH)

    # 2) 読み込めなかった場合は昔ながらのデフォルトを登録
    if not loaded:
        # ここは前に _build_default_llm_manager でやっていた内容をそのまま移植
        manager.register_gpt4o(priority=3.0, enabled=True)
        manager.register_gpt51(priority=2.0, enabled=True)
        manager.register_hermes(priority=1.0, enabled=True)

    _MANAGER_CACHE[key] = manager
    return manager
