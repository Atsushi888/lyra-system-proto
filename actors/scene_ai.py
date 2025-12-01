# actors/scene_ai.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional
import json
import os


@dataclass
class SceneInfo:
    scene_id: str = "town"
    label: str = "街"
    # ボーナス値は Dict[str, float] で保持（affection 等）
    emotion_bonus: Dict[str, float] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "label": self.label,
            "emotion_bonus": self.emotion_bonus or {},
        }


class SceneAI:
    """
    SceneChanger が state に保存した値＋JSON ファイルのボーナス定義をまとめて扱うクラス。
    """

    def __init__(
        self,
        *,
        state: Mapping[str, Any],
        bonus_dir: str = "actors/scene_bonus",
    ) -> None:
        self.state = state
        self.bonus_dir = bonus_dir

    # ---------------------------------------------------------
    def get_scene_info(self) -> SceneInfo:
        scene_id = str(self.state.get("scene_current", "town"))
        label = str(self.state.get("scene_label", "街"))

        bonus = self._load_bonus_for_scene(scene_id)
        return SceneInfo(scene_id=scene_id, label=label, emotion_bonus=bonus)

    def get_emotion_bonus(self) -> Dict[str, float]:
        info = self.get_scene_info()
        return info.emotion_bonus or {}

    # ---------------------------------------------------------
    def _load_bonus_for_scene(self, scene_id: str) -> Dict[str, float]:
        """
        actors/scene_bonus/{scene_id}.json を読み込み、"emotion_bonus" キーを返す。
        存在しない場合は空 dict。
        """
        os.makedirs(self.bonus_dir, exist_ok=True)
        path = os.path.join(self.bonus_dir, f"{scene_id}.json")

        if not os.path.exists(path):
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {}

        bonus = data.get("emotion_bonus")
        if isinstance(bonus, dict):
            # float に寄せて返す
            return {k: float(v) for k, v in bonus.items() if isinstance(v, (int, float))}
        return {}
