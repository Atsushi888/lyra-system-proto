# actors/persona/persona_base/build_emotion_header.py
from __future__ import annotations

from typing import Any, Dict, Optional, List


def _extract_emotion_fields(emotion: Any) -> Dict[str, float]:
    """
    EmotionResult / dict / None をゆるく受け取って数値を引き出す。
    """
    if emotion is None:
        return {
            "affection": 0.0,
            "doki_power": 0.0,
            "doki_level": 0,
            "relationship_level": 0.0,
            "masking_degree": 0.0,
        }

    # EmotionResult 相当のオブジェクト or dataclass を想定
    if hasattr(emotion, "__dict__") and not isinstance(emotion, dict):
        data = emotion.__dict__
    elif isinstance(emotion, dict):
        data = emotion
    else:
        data = {}

    def f(key: str, default: float = 0.0) -> float:
        try:
            return float(data.get(key, default) or default)
        except Exception:
            return default

    def i(key: str, default: int = 0) -> int:
        try:
            return int(data.get(key, default) or default)
        except Exception:
            return default

    return {
        "affection": f("affection", 0.0),
        "doki_power": f("doki_power", 0.0),
        "doki_level": i("doki_level", 0),
        "relationship_level": f("relationship_level", 0.0),
        "masking_degree": f("masking_degree", 0.0),
    }


def build_emotion_header_core(
    *,
    persona: Any,
    emotion: Any | None,
    world_state: Optional[Dict[str, Any]] = None,
    scene_emotion: Optional[Dict[str, Any]] = None,
) -> str:
    """
    system_prompt 直下に足す「感情・関係性ヘッダ」を組み立てる。

    - affection / doki_power / relationship_level / masking_degree
    - world_state（場所・時間・others_present）
    をざっくり可視化する。
    """
    ws = world_state or {}
    se = scene_emotion or {}

    emo_fields = _extract_emotion_fields(emotion)
    affection = emo_fields["affection"]
    doki_power = emo_fields["doki_power"]
    doki_level = int(emo_fields["doki_level"])
    relationship_level = emo_fields["relationship_level"]
    masking_degree = max(0.0, min(emo_fields["masking_degree"], 1.0))

    # 実効好感度（簡易）：ここではそのまま affection を使う
    affection_with_doki = affection

    # 好意の解釈ラベル（あれば Persona 側から）
    try:
        affection_label = persona.get_affection_label(affection_with_doki)
    except Exception:
        affection_label = ""

    # 場所・時間
    locs = ws.get("locations") or {}
    if not isinstance(locs, dict):
        locs = {}
    location = (
        locs.get("player")
        or locs.get("floria")
        or "プレイヤーの部屋"
    )

    t = ws.get("time") or {}
    if not isinstance(t, dict):
        t = {}
    slot_name = t.get("slot") or "morning"
    time_str = t.get("time_str") or "07:30"

    # others_present
    others_raw = ws.get("others_present")
    others_sentence = ""
    if isinstance(others_raw, bool):
        if others_raw:
            others_sentence = (
                "周囲には他の学院生や利用者がいます。"
                "完全な二人きりではないため、振る舞いは控えめに、甘さはささやかに。"
            )
        else:
            # persona.display_name を使ってもいいが、汎用性のため「相手」とぼかす
            others_sentence = (
                "現在、この場には事実上あなたと相手キャラクターだけの二人きりです。"
            )

    # doki_level / mode からのガイドライン（JSON or デフォルト）
    try:
        guideline = persona.build_emotion_control_guideline(
            affection_with_doki=affection_with_doki,
            doki_level=doki_level,
            mode_current=getattr(emotion, "mode", "normal")
            if emotion is not None and hasattr(emotion, "mode")
            else "normal",
        )
    except Exception:
        guideline = persona._build_default_guideline(
            affection_with_doki=affection_with_doki,
            doki_level=doki_level,
            mode_current="normal",
        )

    lines: List[str] = []
    lines.append("[感情・関係性プロファイル]")
    lines.append(
        f"- 実効好感度 (affection_with_doki): {affection_with_doki:.2f} "
        f"(zone=auto, doki_level={doki_level}, doki_power={doki_power:.1f})"
    )

    if affection_label:
        lines.append(f"- 好意の解釈: {affection_label}")

    lines.append(f"- 関係レベル (relationship_level): {relationship_level:.1f} / 100")
    lines.append(
        f"- 表情コントロール（ばけばけ度）: {masking_degree:.2f} "
        "(0=素直 / 1=完全に平静を装う)"
    )

    lines.append(f"- 現在の舞台は「{location}」。")
    lines.append(f"- 時間帯は「{slot_name} / {time_str}」。")

    if others_sentence:
        lines.append(f"- {others_sentence}")
    else:
        lines.append(
            "- 周囲の状況: 特筆すべき外野情報は world_state.others_present に依存します。"
        )

    lines.append("- 備考: ドキドキ💓はその場の高揚感、relationship_level は長期的な信頼・絆の指標です。")
    lines.append("")
    lines.append(guideline)

    return "\n".join(lines)
