# actors/emotion/emotion_state.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


# =========================================
# 関係の長期状態（メモリベース）
# =========================================

@dataclass
class EmotionLongTermState:
    """
    記憶ベースで積算された長期的な関係状態。
    - affection_mean: 長期平均の affection_with_doki（0〜1）
    - relationship_level: 0〜100 のスケール（ゲーム全体の“進行度”用）
    - relationship_stage: テキスト向けのステージ名
    - sample_count: 集計に使われた「記憶レコード」の概算数
    """
    affection_mean: float = 0.0
    relationship_level: float = 0.0
    relationship_stage: str = "acquaintance"
    sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmotionLongTermState":
        if not isinstance(data, dict):
            return cls()
        return cls(
            affection_mean=float(data.get("affection_mean", 0.0) or 0.0),
            relationship_level=float(data.get("relationship_level", 0.0) or 0.0),
            relationship_stage=str(data.get("relationship_stage") or "acquaintance"),
            sample_count=int(data.get("sample_count", 0) or 0),
        )


# =========================================
# 関係レベル → ステージ名
# =========================================

def relationship_stage_from_level(level: float) -> str:
    """
    0〜100 の relationship_level を、ざっくりした段階ラベルに変換する。

    （ゲーム内的な意味合いイメージ）
      0〜10   … acquaintance     : 知り合い〜顔見知り
      10〜30  … friendly         : 友好的／それなりに仲良し
      30〜60  … close_friends    : 親しい友人〜相棒
      60〜85  … dating           : 恋人関係（ほぼ両想い）
      85〜100 … soulmate         : 将来を真剣に考えている相手
    """
    x = max(0.0, min(100.0, level))

    if x >= 85.0:
        return "soulmate"
    if x >= 60.0:
        return "dating"
    if x >= 30.0:
        return "close_friends"
    if x >= 10.0:
        return "friendly"
    return "acquaintance"


# =========================================
# affection → relationship_level 変換
# =========================================

def calc_relationship_level_from_affection(
    affection_long_term: float,
    *,
    current_level: float = 0.0,
    alpha: float = 0.3,
) -> float:
    """
    長期 affection（0〜1）から relationship_level（0〜100）を計算する。

    - affection_long_term: 記憶ベースで平滑化された affection_with_doki
    - current_level: 直前の relationship_level（0〜100）
    - alpha: 反応の速さ（0〜1）。大きいほど最新値に寄せる。

    単純に
        target = affection_long_term * 100
        new = (1 - alpha) * current_level + alpha * target
    という指数平滑のイメージ。
    """
    a = max(0.0, min(1.0, affection_long_term))
    target = a * 100.0

    alpha = max(0.0, min(1.0, alpha))
    new_level = (1.0 - alpha) * float(current_level) + alpha * target
    return max(0.0, min(100.0, new_level))


# =========================================
# ばけばけ度（masking_degree）算出
# =========================================

def calc_masking_degree(
    *,
    relationship_level: float,
    party_mode: str = "alone",
    is_primary_partner: bool = True,
) -> float:
    """
    0〜1 の「表情コントロール度（ばけばけ度）」を返す。

    ざっくり方針:
      - 基本は 0.0〜0.4 程度（あまり盛りすぎない）
      - 人前（party_mode != "alone"）では少し高めにする
      - まだ関係が浅い段階では、逆に“素直度”が下がるように少し上げる

    relationship_level が高く、人前でない & 本命相手 → ほぼ素直（0〜0.1）
    """
    rl = max(0.0, min(100.0, relationship_level))

    # ベースライン：関係が深いほど素直になる（masking が下がる）
    #  - rl=0   → base ≒ 0.4
    #  - rl=100 → base ≒ 0.05
    base = 0.4 - 0.35 * (rl / 100.0)
    base = max(0.05, min(0.4, base))

    # 人前かどうか
    pm = (party_mode or "alone").lower()
    if pm not in ("alone", "private"):
        # クラスメイトがいる／公共空間など → ちょっと上乗せ
        base += 0.25

    # 本命相手なら、がんばって“素直寄り”に戻す
    if is_primary_partner:
        base -= 0.1

    return max(0.0, min(1.0, base))
