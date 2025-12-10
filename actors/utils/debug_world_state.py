# actors/utils/debug_world_state.py
from __future__ import annotations

from typing import Any, Dict, Optional
import json
import os

import streamlit as st


class WorldStateDebugger:
    """
    world_state / scene_emotion / emotion などを、LYRA_DEBUG=1 のときだけ
    まとめてダンプする共通デバッガ。

    world_state には一切手を触れない。
    """

    def __init__(self, name: str = "WorldStateDebugger") -> None:
        self.name = name
        self.enabled: bool = os.getenv("LYRA_DEBUG", "0") == "1"

    def log(
        self,
        *,
        caller: str,
        world_state: Optional[Dict[str, Any]] = None,
        scene_emotion: Optional[Dict[str, Any]] = None,
        emotion: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return

        payload: Dict[str, Any] = {
            "debugger": self.name,
            "caller": caller,
        }
        if world_state is not None:
            payload["world_state"] = world_state
        if scene_emotion is not None:
            payload["scene_emotion"] = scene_emotion
        if emotion is not None:
            payload["emotion"] = emotion
        if extra:
            payload["extra"] = extra

        text = json.dumps(payload, ensure_ascii=False, indent=2)

        header = f"=== [LYRA DEBUG] {self.name} from {caller} ==="
        try:
            st.write(header)
            st.text(text)
        except Exception:
            print(header)
            print(text)
