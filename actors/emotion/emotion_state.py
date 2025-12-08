# actors/emotion/emotion_state.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from actors.emotion.emotion_levels import affection_to_level


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


# =========================================
# EmotionState 本体
# =========================================

@dataclass
class EmotionState:
    """
    MixerAI が統合した「最終的な感情状態」を保持するクラス。

    - EmotionAI の短期結果（llm_meta["emotion"]）
    - 長期状態（llm_meta["emotion_long_term"] などから将来拡張）
    - UI / デバッグ用の上書き（doki スライダー等）

    をまとめて 1 つの dict に正規化し、
    PersonaBase / AnswerTalker / Viewer から参照しやすくする。
    """

    # 短期ベース
    affection: float = 0.0
    affection_with_doki: float = 0.0
    affection_zone: str = "auto"   # low / mid / high / extreme / auto
    doki_power: float = 0.0
    doki_level: int = 0
    mode: str = "normal"           # "normal" / "erotic" など

    # 長期関係
    relationship_level: float = 0.0    # 0〜100
    relationship_stage: str = ""       # acquaintance / friendly / ...

    # ばけばけ度（0=素直 / 1=完全に平静を装う）
    masking_degree: float = 0.0

    # この状態がどこ由来か（debug_dokipower / manual / auto 等）
    source: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # -----------------------------------------
    # 統合ロジック
    # -----------------------------------------
    @classmethod
    def from_sources(
        cls,
        *,
        base: Optional[Dict[str, Any]] = None,
        long_term: Optional[Dict[str, Any]] = None,
        manual: Optional[Dict[str, Any]] = None,
        debug: Optional[Dict[str, Any]] = None,
        source_hint: str = "auto",
    ) -> "EmotionState":
        """
        MixerAI から呼ばれるファクトリ。

        優先度:
          debug > manual > base

        - base: EmotionAI.analyze() の結果（EmotionResult.to_dict 相当）
        - long_term: EmotionAI.update_long_term() の結果（将来拡張用）
        - manual: 将来のUI手動調整用（現時点では未使用）
        - debug: dokipower_control など、強制上書き用
        """
        # -----------------------------
        # どのソースを主として使うか
        # -----------------------------
        src: Dict[str, Any] = {}
        if isinstance(debug, dict) and debug:
            src = debug
            src_origin = "debug_dokipower"
        elif isinstance(manual, dict) and manual:
            src = manual
            src_origin = "manual"
        elif isinstance(base, dict) and base:
            src = base
            src_origin = source_hint or "auto"
        else:
            src_origin = "auto"

        # -----------------------------
        # affection / doki 関連
        # -----------------------------
        aff_raw = float(src.get("affection", 0.0) or 0.0)
        # affection_with_doki があれば優先
        aff_with_doki = float(
            src.get("affection_with_doki", src.get("affection", 0.0)) or 0.0
        )
        aff_with_doki = max(0.0, min(1.0, aff_with_doki))

        doki_power = float(src.get("doki_power", 0.0) or 0.0)
        try:
            doki_level = int(src.get("doki_level", 0) or 0)
        except Exception:
            doki_level = 0
        doki_level = max(0, min(4, doki_level))

        mode = str(src.get("mode") or "normal")

        # affection_zone（JSON 側で計算済みがあれば尊重）
        zone = str(src.get("affection_zone") or "").strip()
        if not zone:
            zone = affection_to_level(aff_with_doki)

        # -----------------------------
        # relationship_level / stage
        # -----------------------------
        # base 側にすでにあればそれを信頼
        rl_from_src = float(src.get("relationship_level", 0.0) or 0.0)

        # long_term 側に明示的な level があれば、補助的に利用
        rl_from_lt = 0.0
        if isinstance(long_term, dict):
            try:
                rl_from_lt = float(long_term.get("relationship_level", 0.0) or 0.0)
            except Exception:
                rl_from_lt = 0.0

        relationship_level = rl_from_src
        if relationship_level <= 0.0 and rl_from_lt > 0.0:
            relationship_level = rl_from_lt

        # どちらにも無ければ、affection からざっくり推定
        if relationship_level <= 0.0 and aff_with_doki > 0.0:
            relationship_level = calc_relationship_level_from_affection(
                affection_long_term=aff_with_doki,
                current_level=0.0,
                alpha=0.5,
            )

        # ステージ名
        rel_stage = str(src.get("relationship_stage") or "").strip()
        if not rel_stage and relationship_level > 0.0:
            rel_stage = relationship_stage_from_level(relationship_level)

        # -----------------------------
        # masking_degree（ばけばけ度）
        # -----------------------------
        if "masking_degree" in src:
            masking_degree = float(src.get("masking_degree") or 0.0)
        elif "masking" in src:
            masking_degree = float(src.get("masking") or 0.0)
        else:
            # world_state / party_mode はここではまだ見ていない。
            # 場所ごとのマスク調整は PersonaBase 側で行う想定。
            masking_degree = calc_masking_degree(
                relationship_level=relationship_level,
                party_mode="alone",
                is_primary_partner=True,
            )

        masking_degree = max(0.0, min(1.0, masking_degree))

        return cls(
            affection=aff_raw,
            affection_with_doki=aff_with_doki,
            affection_zone=zone,
            doki_power=doki_power,
            doki_level=doki_level,
            mode=mode,
            relationship_level=relationship_level,
            relationship_stage=rel_stage,
            masking_degree=masking_degree,
            source=src_origin,
        )
