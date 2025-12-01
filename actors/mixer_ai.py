# actors/mixer_ai.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, MutableMapping

from actors.emotion_ai import EmotionResult
from actors.scene_ai import SceneAI


@dataclass
class EmotionVector:
    """
    感情ベクトルをまとめて扱うための小さなヘルパ。

    keys:
      - affection
      - arousal
      - tension
      - anger
      - sadness
      - excitement
    """
    affection: float = 0.0
    arousal: float = 0.0
    tension: float = 0.0
    anger: float = 0.0
    sadness: float = 0.0
    excitement: float = 0.0

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "EmotionVector":
        if not isinstance(data, dict):
            return cls()
        def f(key: str) -> float:
            v = data.get(key, 0.0)
            try:
                return float(v)
            except Exception:
                return 0.0
        return cls(
            affection=f("affection"),
            arousal=f("arousal"),
            tension=f("tension"),
            anger=f("anger"),
            sadness=f("sadness"),
            excitement=f("excitement"),
        )

    @classmethod
    def from_emotion_result(cls, res: Optional[EmotionResult]) -> "EmotionVector":
        if res is None:
            return cls()
        return cls(
            affection=float(res.affection),
            arousal=float(res.arousal),
            tension=float(res.tension),
            anger=float(res.anger),
            sadness=float(res.sadness),
            excitement=float(res.excitement),
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "affection": self.affection,
            "arousal": self.arousal,
            "tension": self.tension,
            "anger": self.anger,
            "sadness": self.sadness,
            "excitement": self.excitement,
        }

    def add_inplace(self, other: "EmotionVector") -> None:
        self.affection += other.affection
        self.arousal   += other.arousal
        self.tension   += other.tension
        self.anger     += other.anger
        self.sadness   += other.sadness
        self.excitement+= other.excitement


class MixerAI:
    """
    EmotionAI / ユーザー手動入力 / SceneAI の感情値を統合するクラス。

    役割:
      - 「どのソースから感情を持ってくるか」を決める
      - SceneAI からシーン固有のボーナス／ペナルティを取得する
      - 上記をミックスして、ModelsAI2 に渡す emotion_override を生成する
    """

    def __init__(self, state: Optional[MutableMapping[str, Any]] = None) -> None:
        # NOTE: state は必須ではないが、あると SceneAI / EmotionControl と連携しやすい
        self.state = state

    # ----------------------------------------------------------
    # SceneAI からのボーナス取得
    # ----------------------------------------------------------
    def get_scene_emotion_bonus(
        self,
        state: Optional[MutableMapping[str, Any]] = None,
    ) -> EmotionVector:
        """
        SceneAI から「現在シーンに由来する感情ボーナス」を取得して EmotionVector で返す。
        SceneAI 側では、例えば以下のような情報を返す想定:

        {
          "scene_id": "town",
          "label": "街",
          "emotion_bonus": {
              "affection": +0.2,
              "tension":  -0.1,
              ...
          }
        }
        """
        st = state or self.state
        if st is None:
            return EmotionVector()

        try:
            scene_ai = SceneAI(state=st)
            info = scene_ai.get_current_scene_info()
        except Exception:
            return EmotionVector()

        if not isinstance(info, dict):
            return EmotionVector()

        raw_bonus = info.get("emotion_bonus") or {}
        return EmotionVector.from_dict(raw_bonus)

    # ----------------------------------------------------------
    # ModelsAI2 用 emotion_override の構築
    # ----------------------------------------------------------
    def build_emotion_override_for_models(
        self,
        *,
        state: Optional[MutableMapping[str, Any]] = None,
        emotion_ai: Any = None,
    ) -> Optional[Dict[str, Any]]:
        """
        ModelsAI2.collect() に渡す emotion_override を構築する。

        ソース:
          - EmotionAI の短期感情（emotion_ai.last_short_result）
          - ユーザー手動入力（state["emotion_override_manual"]）
          - SceneAI からのボーナス（get_scene_emotion_bonus）

        重み付けルール（v1.1 プロトタイプ）:
          - EmotionAI（base）は 1.0 倍
          - Scene ボーナスはそのまま加算（微調整想定）
          - manual_full の場合:
              手動ベクトルをベースとし、そこに Scene ボーナスだけ足す
        """

        st = state or self.state
        if st is None:
            return None

        # 1) base: EmotionAI の短期感情
        short = getattr(emotion_ai, "last_short_result", None)
        base_vec = EmotionVector.from_emotion_result(short)
        base_mode = getattr(short, "mode", "normal") if short is not None else "normal"

        # 2) user: EmotionControl パネルからの手動入力
        override_mode = st.get("emotion_override_mode", "auto")
        manual_raw = st.get("emotion_override_manual")
        manual_vec = EmotionVector.from_dict(manual_raw) if isinstance(manual_raw, dict) else None

        # 3) scene: SceneAI からのボーナス／ペナルティ
        scene_vec = self.get_scene_emotion_bonus(st)

        # 4) 合成
        if override_mode == "manual_full" and manual_vec is not None:
            # 手動値をベースに、Scene のボーナスだけ足す
            mixed = EmotionVector(
                affection=manual_vec.affection,
                arousal=manual_vec.arousal,
                tension=manual_vec.tension,
                anger=manual_vec.anger,
                sadness=manual_vec.sadness,
                excitement=manual_vec.excitement,
            )
            mode_str = "manual_full"
        else:
            # 通常は EmotionAI の結果をベースに Scene ボーナスを加算
            mixed = EmotionVector(
                affection=base_vec.affection,
                arousal=base_vec.arousal,
                tension=base_vec.tension,
                anger=base_vec.anger,
                sadness=base_vec.sadness,
                excitement=base_vec.excitement,
            )
            mode_str = base_mode

        # Scene ボーナスを加算
        mixed.add_inplace(scene_vec)

        # 何も変化がない（すべて 0）なら None を返してもよいが、
        # ModelsAI2 側で「mode だけでも欲しい」場合があるので常に返す。
        result = {
            "mode": mode_str,
            **mixed.to_dict(),
        }

        # デバッグ用に state にも落としておくと便利
        try:
            st["emotion_mixed_for_models"] = result
        except Exception:
            pass

        return result
