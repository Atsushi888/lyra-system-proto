# actors/emotion/emotion_state.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from actors.emotion_levels import affection_to_level


def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _clamp_int(value: int, lo: int, hi: int) -> int:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def relationship_stage_from_level(level: float) -> str:
    """
    relationship_level (0–100) から、人間可読なステージ説明を返す。

    ※ 仕様（暫定）：
      0–9    : distant … まだほぼ他人
      10–24  : acquaintance … 顔見知り〜知人
      25–39  : friendly … 普通に話せる関係
      40–59  : close_friend … 親友・大事な友人
      60–69  : pre_lover … 事実上の恋人手前（周囲からはそう見える）
      70–84  : lover_full … 完落ちの恋人ゾーン
      85–100 : quasi_married … 夫婦未満・夫婦同然ゾーン
    """
    lv = _clamp(level, 0.0, 100.0)

    if lv >= 85.0:
        return (
            "quasi_married: 夫婦未満・夫婦同然のゾーン。"
            "長期的な生活や家族としての未来を強く意識している。"
        )
    if lv >= 70.0:
        return (
            "lover_full: 恋人として完全に落ち着いているゾーン。"
            "相互の信頼が非常に高く、多少の喧嘩では揺らがない。"
        )
    if lv >= 60.0:
        return (
            "pre_lover: 周囲からは恋人同士と見なされやすいが、"
            "二人の中ではまだ『付き合っている』という言葉を"
            "はっきり交わしていない、甘酸っぱい段階。"
        )
    if lv >= 40.0:
        return (
            "close_friend: 深く信頼し合う親友ゾーン。"
            "どんな相談も打ち明けられるが、恋愛としてはまだ曖昧。"
        )
    if lv >= 25.0:
        return (
            "friendly: 普通に仲の良い友人ゾーン。"
            "一緒にいて楽しいが、特別扱いはまだ少ない。"
        )
    if lv >= 10.0:
        return (
            "acquaintance: 顔見知り〜知人ゾーン。"
            "必要なときに会話はするが、まだ距離感はある。"
        )
    return (
        "distant: ほとんど関わりがない、またはまだ距離がある段階。"
        "丁寧で形式的な対応が中心。"
    )


@dataclass
class EmotionState:
    """
    MixerAI が統合した「そのターンの感情状態」の標準フォーマット。

    - affection / arousal 系 … 短期的な感情（従来の EmotionResult に近い）
    - doki_power / doki_level … 好きな相手を前にした「高揚度」の指標（短期）
    - relationship_level      … 長期的な関係の深さ（0〜100）
    - masking_degree          … ばけばけ度。0=素直 / 1=完全に平静を装う
    """

    # 短期感情
    mode: str = "normal"
    affection: float = 0.0
    arousal: float = 0.0
    tension: float = 0.0
    anger: float = 0.0
    sadness: float = 0.0
    excitement: float = 0.0

    # ドキドキ系
    doki_power: float = 0.0  # 0〜100
    doki_level: int = 0      # 0〜4

    # 長期関係
    relationship_level: float = 0.0  # 0〜100
    relationship_stage: str = ""

    # ばけばけ度
    masking_degree: float = 0.0      # 0.0〜1.0

    # 派生値
    affection_with_doki: float = 0.0
    affection_zone: str = "low"      # low / mid / high / extreme

    # ソース情報（デバッグ用）
    source: str = "auto"             # "auto" / "debug_dokipower" など

    # -----------------------------
    # 生成ユーティリティ
    # -----------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmotionState":
        """
        dict（従来の EmotionResult.to_dict() や emotion_long_term）から生成。
        未指定項目はデフォルト値。
        """
        d = dict(data or {})

        mode = str(d.get("mode", "normal") or "normal")

        affection = float(d.get("affection", 0.0) or 0.0)
        arousal = float(d.get("arousal", 0.0) or 0.0)
        tension = float(d.get("tension", 0.0) or 0.0)
        anger = float(d.get("anger", 0.0) or 0.0)
        sadness = float(d.get("sadness", 0.0) or 0.0)
        excitement = float(d.get("excitement", 0.0) or 0.0)

        doki_power = float(d.get("doki_power", 0.0) or 0.0)
        doki_level = _clamp_int(int(d.get("doki_level", 0) or 0), 0, 4)

        # relationship は long_term 側や debug から入る想定
        relationship_level = float(
            d.get("relationship_level", d.get("relationship", 0.0)) or 0.0
        )
        relationship_level = _clamp(relationship_level, 0.0, 100.0)

        relationship_stage = str(d.get("relationship_stage", "") or "")

        masking_degree = float(
            d.get("masking_degree", d.get("masking", 0.0)) or 0.0
        )
        masking_degree = _clamp(masking_degree, 0.0, 1.0)

        # affection_with_doki は既存値を優先。無ければ簡易計算。
        awd_raw = d.get("affection_with_doki", None)
        if awd_raw is None:
            # 簡易版：doki_power(0-100) を最大 +0.3 相当にスケーリング
            bonus = _clamp(doki_power, 0.0, 100.0) / 100.0 * 0.3
            affection_with_doki = _clamp(affection + bonus, 0.0, 1.0)
        else:
            affection_with_doki = _clamp(float(awd_raw or 0.0), 0.0, 1.0)

        # affection_zone は従来ロジックに委ねる
        affection_zone = d.get("affection_zone") or affection_to_level(
            affection_with_doki
        )

        # source は後から上書きされることが多いので最低限
        source = str(d.get("source", "auto") or "auto")

        return cls(
            mode=mode,
            affection=affection,
            arousal=arousal,
            tension=tension,
            anger=anger,
            sadness=sadness,
            excitement=excitement,
            doki_power=doki_power,
            doki_level=doki_level,
            relationship_level=relationship_level,
            relationship_stage=relationship_stage,
            masking_degree=masking_degree,
            affection_with_doki=affection_with_doki,
            affection_zone=affection_zone,
            source=source,
        )

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
        複数ソース（base / long_term / manual / debug）をマージして EmotionState を生成。

        優先度（下から上に向かって上書き）:
            long_term  <  base(EmotionAI)  <  manual  <  debug(dokipower)
        """
        merged: Dict[str, Any] = {}
        for src in (long_term, base, manual, debug):
            if isinstance(src, dict):
                merged.update(src)

        state = cls.from_dict(merged)

        if debug:
            state.source = "debug_dokipower"
        else:
            state.source = source_hint or "auto"

        # relationship_stage が空ならここで補完
        if not state.relationship_stage:
            state.relationship_stage = relationship_stage_from_level(
                state.relationship_level
            )

        # affection_zone も念のため更新
        state.affection_zone = affection_to_level(state.affection_with_doki)

        return state

    # -----------------------------
    # 出力
    # -----------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        MixerAI → emotion_override["emotion"] へ入れるための dict 形式。
        従来の EmotionResult dict と互換性を保ちつつ、関係値も含める。
        """
        data = asdict(self)
        # dataclass のキーのままで問題ないが、念のため zone も追加
        if "affection_zone" not in data or not data["affection_zone"]:
            data["affection_zone"] = affection_to_level(self.affection_with_doki)

        return data
