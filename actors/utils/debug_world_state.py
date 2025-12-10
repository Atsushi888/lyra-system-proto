# actors/utils/debug_world_state.py
from __future__ import annotations

from typing import Any, Dict, Optional
import json
import os

import streamlit as st


class WorldStateDebugger:
    """
    world_state / scene_emotion / emotion などを、
    環境変数 LYRA_DEBUG=1 のときだけまとめて可視化するための小さなデバッガ。
    """

    def __init__(self, name: str = "WorldStateDebugger") -> None:
        self.name = name
        # LYRA_DEBUG=1 のときだけ有効
        self.enabled: bool = os.getenv("LYRA_DEBUG", "0") == "1"

    def log(
        self,
        *,
        caller: str,
        step: Optional[str] = None,
        world_state: Optional[Dict[str, Any]] = None,
        scene_emotion: Optional[Dict[str, Any]] = None,
        emotion: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        各所からのデバッグ情報をまとめて出力する。

        Parameters
        ----------
        caller : 呼び出し元の識別名（例: "MixerAI.build_emotion_override"）
        step   : 任意のステップ名（"after_merge" 等）
        world_state, scene_emotion, emotion, extra :
                 それぞれ任意の dict を渡す。None のものは省略される。
        """
        if not self.enabled:
            return

        payload: Dict[str, Any] = {
            "debugger": self.name,
            "caller": caller,
        }

        if step is not None:
            payload["step"] = step

        if world_state is not None:
            payload["world_state"] = world_state
        if scene_emotion is not None:
            payload["scene_emotion"] = scene_emotion
        if emotion is not None:
            payload["emotion"] = emotion
        if extra is not None:
            payload["extra"] = extra

        try:
            st.markdown("##### === [LYRA DEBUG] WorldStateDebugger ===")
            st.json(payload)
        except Exception:
            # Streamlit が使えない環境向けフォールバック
            print(
                "[LYRA DEBUG] WorldStateDebugger:",
                json.dumps(payload, ensure_ascii=False, indent=2),
            )


# グローバルなシングルトン + 薄いラッパ
_world_debugger = WorldStateDebugger()


def debug_world_state(
    *,
    caller: str,
    step: Optional[str] = None,
    world_state: Optional[Dict[str, Any]] = None,
    scene_emotion: Optional[Dict[str, Any]] = None,
    emotion: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    各所から気軽に呼ぶための簡易ヘルパ。

    例:
        debug_world_state(
            caller="MixerAI.build_emotion_override",
            step="after_merge",
            world_state=world_state,
            scene_emotion=scene_emotion,
            emotion=emotion,
            extra={"has_emo_manual": bool(emo_manual)},
        )
    """
    _world_debugger.log(
        caller=caller,
        step=step,
        world_state=world_state,
        scene_emotion=scene_emotion,
        emotion=emotion,
        extra=extra,
    )
