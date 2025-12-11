# actors/persona/persona_base/build_emotion_header.py
from __future__ import annotations

from typing import Any, Dict, Optional

from actors.emotion_ai import EmotionResult
from actors.persona.persona_base.persona_base import PersonaBase


def build_emotion_header_core(
    *,
    persona: PersonaBase,
    emotion: Optional[EmotionResult],
    world_state: Optional[Dict[str, Any]],
    scene_emotion: Optional[Dict[str, Any]],
) -> str:
    """
    旧 API 用の簡易ヘッダ構築。
    新設の build_emotion_based_system_prompt_core よりも情報量は少なめだが、
    互換性維持のために残しておく。
    """
    if emotion is None:
        return ""

    ws = world_state or {}
    loc = ws.get("locations") or {}
    if not isinstance(loc, dict):
        loc = {}
    location = loc.get("player") or loc.get("floria") or "プレイヤーの部屋"

    t = ws.get("time") or {}
    if not isinstance(t, dict):
        t = {}
    slot = t.get("slot", "morning")
    time_str = t.get("time_str", "07:30")

    others_present_raw = ws.get("others_present")
    others_present: Optional[bool] = None
    if isinstance(others_present_raw, bool):
        others_present = others_present_raw

    aff_with_doki = getattr(emotion, "affection_with_doki", emotion.affection)
    doki_level = getattr(emotion, "doki_level", 0)
    relationship_level = getattr(emotion, "relationship_level", 0.0)
    masking_degree = getattr(emotion, "masking_degree", 0.0)

    aff_label = persona.get_affection_label(aff_with_doki) or ""

    lines: list[str] = []
    lines.append("[感情ヘッダ（互換用）]")
    lines.append(f"- affection_with_doki: {aff_with_doki:.2f}")
    if aff_label:
        lines.append(f"- 好意の解釈: {aff_label}")
    lines.append(f"- doki_level: {int(doki_level)}")
    lines.append(f"- relationship_level: {relationship_level:.1f}")
    lines.append(
        f"- masking_degree: {masking_degree:.2f} "
        "(0=素直 / 1=完全に平静を装う)"
    )
    lines.append(f"- 現在の舞台: {location}")
    lines.append(f"- 時間帯: {slot} / {time_str}")

    if others_present is True:
        lines.append("- 周囲に他の人がいるため、感情表現はやや控えめに。")
    elif others_present is False:
        lines.append("- 現在は二人きりなので、少しだけ素直な感情を見せてもよい。")

    # JSON がない場合のフォールバック・ガイドライン
    lines.append("")
    lines.append(
        persona._build_default_guideline(
            affection_with_doki=aff_with_doki,
            doki_level=int(doki_level),
            mode_current=getattr(emotion, "mode", "normal"),
        )
    )

    return "\n".join(lines)
