from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from actors.emotion_ai import EmotionAI, EmotionResult
from actors.scene_ai import SceneAI


@dataclass
class MixedEmotion:
    """
    EmotionAI（短期）＋手動オーバーライド＋SceneAI ボーナスをマージした結果。
    """
    mode: str = "normal"
    affection: float = 0.0
    arousal: float = 0.0
    tension: float = 0.0
    anger: float = 0.0
    sadness: float = 0.0
    excitement: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "affection": self.affection,
            "arousal": self.arousal,
            "tension": self.tension,
            "anger": self.anger,
            "sadness": self.sadness,
            "excitement": self.excitement,
        }


class MixerAI:
    """
    感情値ミキサー。

    - base: EmotionAI の短期感情（EmotionResult）
    - manual: 手動オーバーライドパネル（state["emotion_override_manual"]）
    - scene: SceneAI からのボーナス（場所ごとの補正）

    を混ぜ合わせて ModelsAI2 に渡す emotion_override 辞書を作る。
    """

    def __init__(
        self,
        *,
        state: Mapping[str, Any],
        emotion_ai: EmotionAI,
        scene_ai: SceneAI,
    ) -> None:
        self.state = state
        self.emotion_ai = emotion_ai
        self.scene_ai = scene_ai

    # ---------------------------------------------------------
    # public API
    # ---------------------------------------------------------
    def build_emotion_override(self) -> Optional[Dict[str, Any]]:
        """
        ModelsAI2.collect() に渡す emotion_override を組み立てる。

        emotion_override_mode:
          - "auto"        : EmotionAI の短期感情＋SceneAI ボーナス
          - "manual_full" : 手動パネルの値で完全上書き（SceneAI も無視）
        """
        mode = str(self.state.get("emotion_override_mode", "auto"))
        manual = self.state.get("emotion_override_manual")

        # 1) manual_full モードなら完全上書き
        if mode == "manual_full" and isinstance(manual, dict):
            return self._from_manual(manual)

        # 2) 通常（auto）モード
        base_short: Optional[EmotionResult] = getattr(
            self.emotion_ai, "last_short_result", None
        )
        if base_short is None:
            # 感情解析がまだ行われていない場合は None
            return None

        return self._from_auto(base_short)

    # ---------------------------------------------------------
    # internal helpers
    # ---------------------------------------------------------
    @staticmethod
    def _from_manual(manual: Dict[str, Any]) -> Dict[str, Any]:
        m = MixedEmotion(
            mode=str(manual.get("mode", "normal")),
            affection=float(manual.get("affection", 0.0)),
            arousal=float(manual.get("arousal", 0.0)),
            tension=float(manual.get("tension", 0.0)),
            anger=float(manual.get("anger", 0.0)),
            sadness=float(manual.get("sadness", 0.0)),
            excitement=float(manual.get("excitement", 0.0)),
        )
        return m.to_dict()

    def _from_auto(self, base: EmotionResult) -> Dict[str, Any]:
        # まず EmotionAI の結果をベースにする
        mixed = MixedEmotion(
            mode=base.mode or "normal",
            affection=float(base.affection),
            arousal=float(base.arousal),
            tension=float(base.tension),
            anger=float(base.anger),
            sadness=float(base.sadness),
            excitement=float(base.excitement),
        )

        # SceneAI からボーナス（存在しないならゼロ扱い）
        scene_bonus = self.scene_ai.get_scene_emotion() or {}
        mixed.affection += float(scene_bonus.get("affection", 0.0))
        mixed.arousal += float(scene_bonus.get("arousal", 0.0))
        mixed.tension += float(scene_bonus.get("tension", 0.0))
        mixed.anger += float(scene_bonus.get("anger", 0.0))
        mixed.sadness += float(scene_bonus.get("sadness", 0.0))
        mixed.excitement += float(scene_bonus.get("excitement", 0.0))

        # 必要ならクランプ（-3.0〜+3.0 など）を入れてもよい
        return mixed.to_dict()
