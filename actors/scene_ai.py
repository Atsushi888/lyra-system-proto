# actors/scene_ai.py
from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional

try:
    import streamlit as st  # Streamlit 環境ならこれを使う
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]


class SceneAI:
    """
    SceneChanger が st.session_state に書き込んだシーン情報を読み取り、
    AnswerTalker / MixerAI が使いやすい dict 形式に整えて返すだけの軽量クラス。

    想定する state キー（prefix = "scene_" の場合）:
      - "scene_current": シーンID 例) "town", "road", "ice_cave", "end"
      - "scene_label":  表示名 例) "街", "街道筋"
      - "scene_emotion_bonus": Dict[str, float]
          {affection, arousal, tension, anger, sadness, excitement}
    """

    EMOTION_KEYS = [
        "affection",
        "arousal",
        "tension",
        "anger",
        "sadness",
        "excitement",
    ]

    def __init__(
        self,
        *,
        state: Optional[MutableMapping[str, Any]] = None,
        prefix: str = "scene_",
    ) -> None:
        # Streamlit ありなら session_state をデフォルトに
        if state is not None:
            self.state = state
        elif st is not None:
            self.state = st.session_state  # type: ignore[assignment]
        else:
            # 非Streamlit環境用の簡易 Dict
            self.state = {}  # type: ignore[assignment]

        self.prefix = prefix

    # ---------------------------------------------------------
    # 内部ヘルパ
    # ---------------------------------------------------------
    def _get(self, key: str, default: Any = None) -> Any:
        return self.state.get(f"{self.prefix}{key}", default)

    # ---------------------------------------------------------
    # 公開 API
    # ---------------------------------------------------------
    def get_scene_info(self) -> Dict[str, Any]:
        """
        現在のシーン情報を dict で返す。

        戻り値:
            {
              "scene_id": str,
              "label": str,
              "emotion_bonus": {affection..excitement}
            }
        """
        scene_id = str(self._get("current", "town"))
        label = str(self._get("label", scene_id))

        raw_bonus = self._get("emotion_bonus") or {}
        bonus: Dict[str, float] = {}
        for k in self.EMOTION_KEYS:
            v = raw_bonus.get(k, 0.0) if isinstance(raw_bonus, dict) else 0.0
            try:
                bonus[k] = float(v)
            except Exception:
                bonus[k] = 0.0

        return {
            "scene_id": scene_id,
            "label": label,
            "emotion_bonus": bonus,
        }
