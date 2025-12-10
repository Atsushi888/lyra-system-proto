# actors/utils/debug_world_state.py
from __future__ import annotations

from typing import Any, Dict, Optional
import json
import os

try:
    import streamlit as st  # type: ignore
except Exception:
    st = None


class WorldStateDebugger:
    """
    world_state / emotion_override をまとめて可視化する専用デバッガ。

    - 環境変数 LYRA_DEBUG == "1" のときだけ出力する
    - caller 文字列で「どこから呼ばれたか」を明示する
    - Streamlit 環境なら st.write/st.code、それ以外なら print にフォールバック
    """

    def __init__(self, name: str = "WorldStateDebugger") -> None:
        self.name = name
        self.debug_enabled: bool = os.getenv("LYRA_DEBUG", "0") == "1"

    def log(
        self,
        *,
        caller: str,
        world_state: Optional[Dict[str, Any]] = None,
        emotion_override: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        caller: どこから呼ばれたか（"MixerAI.build_emotion_override" など）
        world_state: SceneAI / DokiPowerControl が持っている world_state 丸ごと
        emotion_override: MixerAI が組み立てた payload（あれば）
        extra: 任意の追加情報（llm_meta の一部など）
        """
        if not self.debug_enabled:
            return

        payload: Dict[str, Any] = {
            "caller": caller,
            "world_state": world_state,
        }
        if emotion_override is not None:
            payload["emotion_override"] = emotion_override
        if extra:
            payload["extra"] = extra

        text = json.dumps(payload, ensure_ascii=False, indent=2)

        # Streamlit が使えるならそっち優先
        if st is not None:
            try:
                st.write(f"[DEBUG:{self.name}] {caller}")
            except Exception:
                pass

            try:
                st.code(text, language="json")
            except Exception:
                # code が死んでもテキストだけは出す
                try:
                    st.write(text)
                except Exception:
                    pass
        else:
            # CLI / テスト環境など
            print(f"[DEBUG:{self.name}] {caller}")
            print(text)
