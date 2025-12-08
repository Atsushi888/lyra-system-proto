# actors/emotion/emotion_models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from actors.emotion_ai import EmotionResult
from actors.emotion.emotion_levels import affection_to_level


@dataclass
class EmotionModel:
    """
    EmotionResult をラップして、
    - affection_with_doki の実効値
    - affection のゾーン（low/mid/high/extreme）
    - relationship_stage / relationship_label / relationship_level
    - masking_degree（ばけばけ度）
    などの「振る舞い」をまとめるクラス。

    まずは AnswerTalker から軽く使うだけの薄いラッパとして実装。
    将来的に decide_judge_mode などもここに寄せていける。
    """

    result: EmotionResult

    # ==============================
    # 便利プロパティ
    # ==============================
    @property
    def affection_effective(self) -> float:
        """
        affection_with_doki が設定されていればそれを優先、
        なければ素の affection を返す。
        """
        if self.result.affection_with_doki > 0:
            return self.result.affection_with_doki
        return self.result.affection

    @property
    def affection_zone(self) -> str:
        """
        affection_effective を 0.0〜1.0 として、
        low / mid / high / extreme のどれかを返す。
        """
        return affection_to_level(self.affection_effective)

    @property
    def is_doki_active(self) -> bool:
        """
        一時的なドキドキが「有意」と見なせるかどうか。
        いまはシンプルに doki_level / doki_power のどちらかが立っていれば True。
        """
        return (self.result.doki_level > 0) or (self.result.doki_power > 0.1)

    # ==============================
    # relationship_stage / label / level / masking
    # ==============================
    def compute_relationship_stage(self) -> int:
        """
        affection_effective を元に、ざっくり relationship_stage を決める。

        仮ルール（必要に応じてチューニング可）:
          0.0〜0.19: 0 (neutral / classmates)
          0.2〜0.39: 1 (friends)
          0.4〜0.59: 2 (close friends / budding romance)
          0.6〜0.79: 3 (dating)
          0.8〜1.0: 4 (engaged / nearly married)
        """
        a = self.affection_effective

        if a >= 0.8:
            return 4
        if a >= 0.6:
            return 3
        if a >= 0.4:
            return 2
        if a >= 0.2:
            return 1
        return 0

    def compute_relationship_label(self, stage: Optional[int] = None) -> str:
        """
        relationship_stage に対応するラベル文字列を返す。
        """
        s = self.result.relationship_stage if stage is None else stage
        mapping = {
            0: "neutral",
            1: "friends",
            2: "close_friends",
            3: "dating",
            4: "engaged",
        }
        return mapping.get(s, "neutral")

    def compute_relationship_level(self) -> float:
        """
        relationship_level（0〜100）の標準定義。

        まずは affection_effective を単純に 0〜100 にスケール。
        """
        level = self.affection_effective * 100.0
        if level < 0.0:
            level = 0.0
        if level > 100.0:
            level = 100.0
        return level

    def compute_masking_degree(self, level: Optional[float] = None) -> float:
        """
        ばけばけ度（0〜1）を計算する。

        デフォルト実装:
          - relationship_level が高いほど「隠さない」＝ masking は小さく
          - 0 のとき 1.0（完全に隠す）、100 のとき 0.0（まったく隠さない）
        """
        if level is None:
            level = self.result.relationship_level

        try:
            lv = float(level or 0.0)
        except Exception:
            lv = 0.0

        if lv < 0.0:
            lv = 0.0
        if lv > 100.0:
            lv = 100.0

        masking = 1.0 - (lv / 100.0)
        if masking < 0.0:
            masking = 0.0
        if masking > 1.0:
            masking = 1.0
        return masking

    def sync_relationship_fields(self) -> None:
        """
        現在の affection_effective から relationship_stage / label /
        relationship_level / masking_degree を再計算し、
        EmotionResult 側のフィールドへ反映する。
        """
        stage = self.compute_relationship_stage()
        label = self.compute_relationship_label(stage)
        level = self.compute_relationship_level()
        masking = self.compute_masking_degree(level)

        self.result.relationship_stage = stage
        self.result.relationship_label = label
        self.result.relationship_level = level
        self.result.masking_degree = masking

    # ==============================
    # judge_mode 補助（今はまだ薄め）
    # ==============================
    def decide_judge_mode(self, current_mode: str = "normal") -> str:
        """
        将来的に『感情状態に応じて judge_mode を切り替える』ためのフック。

        いまは挙動を変えずに current_mode をそのまま返す。
        （本格運用するタイミングでロジックを追加していけばOK）
        """
        return current_mode

    # ==============================
    # Mixer / Prompt 用 payload
    # ==============================
    def to_override_emotion_dict(self) -> Dict[str, Any]:
        """
        MixerAI や PersonaBase に渡すことを想定した
        emotion_override["emotion"] 相当の dict を返す。
        """
        return self.result.to_dict()

    def to_debug_snapshot(self) -> Dict[str, Any]:
        """
        AnswerTalkerView などでデバッグ表示しやすい形のスナップショット。
        """
        return {
            "affection_effective": self.affection_effective,
            "affection_zone": self.affection_zone,
            "relationship_stage": self.result.relationship_stage,
            "relationship_label": self.result.relationship_label,
            "relationship_level": self.result.relationship_level,
            "masking_degree": self.result.masking_degree,
            "doki_level": self.result.doki_level,
            "doki_power": self.result.doki_power,
        }
