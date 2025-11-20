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


def get_llm_manager(persona_id: str) -> LLMManager:
    key = f"llm_manager_{persona_id}"

    # 既に作ってあれば、それを返す
    if key in st.session_state:
        return st.session_state[key]

    # 新規作成
    manager = LLMManager(persona_id=persona_id)

    # LLM デフォルト設定ロード
    loaded = manager.load_default_config()

    # 読み込みに失敗した場合は、自動でデフォルト登録
    if not loaded:
        manager.register_gpt4o(priority=3.0, enabled=True)
        manager.register_gpt51(priority=2.0, enabled=True)
        manager.register_hermes(priority=1.0, enabled=True)

    # セッションに保存
    st.session_state[key] = manager
    return manager
