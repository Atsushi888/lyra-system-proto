# actors/utils/others_present_helper.py
from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def resolve_others_present(world_state: Dict[str, Any] | None) -> bool:
    """
    「このシーンにプレイヤー＋相手キャラ以外の生徒がいるか？」を
    world_state / manual_controls / emotion_manual_controls から決定する。

    - 最優先: world_state.flags.others_present (bool)
    - 次点:   world_state["others_present"] (bool)
    - 次点:   world_state_manual_controls["others_present"] (bool)
    - 次点:   emotion_manual_controls["others_present"] (bool)
              or emotion_manual_controls["environment"] == "with_others"
    - どこにも無ければ False（= 二人きり扱い）
    """
    ws = world_state or {}
    if not isinstance(ws, dict):
        ws = {}

    # 1) world_state.flags.others_present
    flags = ws.get("flags")
    if isinstance(flags, dict):
        v = flags.get("others_present")
        if isinstance(v, bool):
            return v

    # 2) world_state["others_present"]
    v = ws.get("others_present")
    if isinstance(v, bool):
        return v

    # 3) world_state_manual_controls
    manual_ws = st.session_state.get("world_state_manual_controls") or {}
    if isinstance(manual_ws, dict):
        v = manual_ws.get("others_present")
        if isinstance(v, bool):
            return v

    # 4) emotion_manual_controls
    manual_emo = st.session_state.get("emotion_manual_controls") or {}
    if isinstance(manual_emo, dict):
        v = manual_emo.get("others_present")
        if isinstance(v, bool):
            return v
        env = manual_emo.get("environment")
        if env == "with_others":
            return True
        if env == "alone":
            return False

    # 5) デフォルト：外野なし（二人きり）
    return False
