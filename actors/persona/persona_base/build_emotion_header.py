# actors/persona/persona_base/build_emotion_header.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from actors.emotion_ai import EmotionResult
from actors.emotion.emotion_levels import affection_to_level


def build_emotion_header_core(
    persona: Any,
    emotion: EmotionResult | None,
    world_state: Dict[str, Any] | None = None,
    scene_emotion: Dict[str, Any] | None = None,
) -> str:
    """
    PersonaBase.build_emotion_header の本体。

    persona は PersonaBase 互換インスタンスを想定：
      - get_emotion_profile()
      - get_affection_label()
      - build_emotion_control_guideline()
    などを持っている必要がある。
    """
    if emotion is None:
        return ""

    world_state = world_state or {}
    scene_emotion = scene_emotion or {}

    # 1) サブクラス完全オーバーライドがあれば優先
    if hasattr(persona, "build_emotion_header_hint"):
        try:
            custom = persona.build_emotion_header_hint(
                emotion=emotion,
                world_state=world_state,
                scene_emotion=scene_emotion,
            )
            if isinstance(custom, str) and custom.strip():
                return custom.strip()
        except Exception:
            pass

    # 2) Persona プロファイルから係数取得
    aff_gain = 1.0
    doki_bias = 0.0
    try:
        prof = persona.get_emotion_profile() or {}
        aff_gain = float(prof.get("affection_gain", 1.0) or 1.0)
        doki_bias = float(prof.get("doki_bias", 0.0) or 0.0)
    except Exception:
        pass

    # 3) affection_with_doki * gain を 0〜1 にクランプ
    base_aff = float(getattr(emotion, "affection", 0.0) or 0.0)
    aff_with_doki_raw = float(
        getattr(emotion, "affection_with_doki", base_aff) or base_aff
    )
    aff = max(0.0, min(1.0, aff_with_doki_raw * aff_gain))

    # 4) doki_level 0〜4 + bias → [0,4] クランプ
    try:
        dl_raw = int(getattr(emotion, "doki_level", 0) or 0)
    except Exception:
        dl_raw = 0
    dl = int(round(dl_raw + doki_bias))
    if dl < 0:
        dl = 0
    if dl > 4:
        dl = 4

    # 5) affection のゾーン（low/mid/high/extreme）
    aff_zone = affection_to_level(aff)

    # 6) 好意ラベル（あれば）
    affection_label = persona.get_affection_label(aff)

    # 7) world_state → 環境ヒント
    location = (
        world_state.get("location_name")
        or world_state.get("player_location")
        or (world_state.get("locations") or {}).get("player")
    )
    time_slot = (
        world_state.get("time_slot")
        or world_state.get("time_of_day")
        or (world_state.get("time") or {}).get("slot")
    )

    scene_hint_parts: List[str] = []
    if location:
        scene_hint_parts.append(f"いま二人は『{location}』付近にいます。")
    if time_slot:
        scene_hint_parts.append(f"時間帯は『{time_slot}』頃です。")
    scene_hint = " ".join(scene_hint_parts).strip()

    # 8) ガイドライン（JSON 優先 / なければ簡易デフォルト）
    try:
        guideline_text = persona.build_emotion_control_guideline(
            affection_with_doki=aff,
            doki_level=dl,
            mode_current=getattr(emotion, "mode", "normal"),
        )
    except Exception:
        guideline_text = ""

    if not guideline_text:
        guideline_lines = [
            "[口調・距離感のガイドライン]",
            "1) 特別な感情プロファイルが未設定のため、通常時と同様のトーンで話してください。",
            "2) 相手への基本的な信頼や好意は感じられるように、やわらかな言葉選びを心がけてください。",
        ]
        guideline_text = "\n".join(guideline_lines)

    guideline_text = guideline_text.strip("\n")

    # 9) ヘッダ構築
    header_lines: List[str] = []
    header_lines.append("【感情・関係性プロファイル】")
    header_lines.append(
        f"- 実効好感度（affection_with_doki）: {aff:.2f} "
        f"(zone={aff_zone}, doki_level={dl})"
    )
    if affection_label:
        header_lines.append(f"- 好意の解釈: {affection_label}")
    if scene_hint:
        header_lines.append(f"- 環境: {scene_hint}")

    header_lines.append("")
    header_block = "\n".join(header_lines)

    return header_block + "\n\n" + guideline_text + "\n"
