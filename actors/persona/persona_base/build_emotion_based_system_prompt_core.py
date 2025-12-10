# actors/persona/build_emotion_based_system_prompt_core.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from actors.utils.debug_world_state import WorldStateDebugger

WS_DEBUGGER = WorldStateDebugger(name="PromptCore")


def build_emotion_based_system_prompt_core(
    persona: Any,
    *,
    base_system_prompt: str,
    emotion_override: Optional[Dict[str, Any]] = None,
    mode_current: str = "normal",
    length_mode: str = "auto",
) -> str:
    emotion_override = emotion_override or {}
    world_state = emotion_override.get("world_state") or {}
    scene_emotion = emotion_override.get("scene_emotion") or {}
    emotion = emotion_override.get("emotion") or {}

    # 🔍 ここだけ追加：world_state をダンプする（読み取り専用）
    WS_DEBUGGER.log(
        caller="build_emotion_based_system_prompt_core",
        world_state=world_state,
        scene_emotion=scene_emotion,
        emotion=emotion,
        extra={"mode_current": mode_current, "length_mode": length_mode},
    )

    # ↓↓↓ ここから下は、**今あなたが貼ってくれた元コードをそのまま使う** ↓↓↓
    # affection は doki 補正後を優先
    affection = float(
        emotion.get("affection_with_doki", emotion.get("affection", 0.0)) or 0.0
    )
    doki_power = float(emotion.get("doki_power", 0.0) or 0.0)
    doki_level = int(emotion.get("doki_level", 0) or 0)

    zone = str(emotion.get("affection_zone", "auto") or "auto")

    relationship_level = float(
        emotion.get("relationship_level", emotion.get("relationship", 0.0)) or 0.0
    )
    relationship_stage = str(emotion.get("relationship_stage") or "")
    if not relationship_stage and relationship_level > 0.0:
        from actors.emotion.emotion_state import relationship_stage_from_level
        relationship_stage = relationship_stage_from_level(relationship_level)

    masking_degree = float(
        emotion.get("masking_degree", emotion.get("masking", 0.0)) or 0.0
    )
    if masking_degree < 0.0:
        masking_degree = 0.0
    if masking_degree > 1.0:
        masking_degree = 1.0

    loc_player = (world_state.get("locations") or {}).get("player")
    location_name = (
        loc_player
        or world_state.get("location_name")
        or world_state.get("player_location")
    )
    time_info = world_state.get("time") or {}
    time_slot = time_info.get("slot") or world_state.get("time_of_day")
    time_str = time_info.get("time_str")

    party_mode = (
        world_state.get("party_mode")
        or (world_state.get("party") or {}).get("mode")
    )
    others_around_flag = world_state.get("others_around")

    if isinstance(others_around_flag, bool):
        others_around: Optional[bool] = others_around_flag
    else:
        if party_mode in ("both", "others", "group"):
            others_around = True
        elif party_mode == "alone":
            others_around = False
        else:
            others_around = None

    is_alone = False
    if party_mode == "alone":
        is_alone = True
    if others_around is False:
        is_alone = True
    if others_around is True:
        is_alone = False

    masking_cfg = persona._get_masking_defaults()
    unmasked_locs = masking_cfg.get("unmasked_locations", [])
    masked_locs = masking_cfg.get("masked_locations", [])

    loc_key = str(location_name or "").lower()
    is_unmasked_place = bool(loc_key and any(tag in loc_key for tag in unmasked_locs))
    is_masked_place = bool(loc_key and any(tag in loc_key for tag in masked_locs))

    masking_env_note = ""
    example_line = ""
    rules = masking_cfg.get("rules") or {}
    raw_example = rules.get("example_line")
    if isinstance(raw_example, str) and raw_example.strip():
        example_line = raw_example.replace("{PLAYER_NAME}", persona.player_name)

    if is_unmasked_place:
        masking_env_note = (
            "※ 現在は親しい相手とくつろげる場所にいるため、"
            "表情コントロール（ばけばけ度）があってもほとんど働かず、"
            "素直なデレや甘えがそのまま表に出て構いません。"
        )
        if example_line:
            masking_env_note += f"\n  例: 「{example_line}」"
    elif is_masked_place:
        if is_alone:
            masking_env_note = (
                "※ 形式上は人目のある場所ですが、いまは実質二人きりなので、"
                "ばけばけ度はあまり気にせず素直な恋愛感情を見せて構いません。"
            )
            if example_line:
                masking_env_note += f"\n  例: 「{example_line}」"
        else:
            masking_env_note = (
                "※ ここは人目のある場所のため、"
                "ばけばけ度を意識して外見上は一段階落ち着いたトーンで振る舞ってください。"
                "内心のドキドキや恋愛感情は、仕草や視線、ささやかな言葉ににじませる程度に留めてください。"
            )

    # ===== others_present の決定ロジック（元のまま） =====
    others_present_flag: bool | None = None
    if isinstance(world_state, dict) and "others_present" in world_state:
        raw_flag = world_state.get("others_present")
        if isinstance(raw_flag, bool):
            others_present_flag = raw_flag

    if others_present_flag is None:
        if others_around is True or party_mode in ("others", "group"):
            others_present_flag = True
        elif is_alone:
            others_present_flag = False
        elif party_mode == "both":
            others_present_flag = False

    location_lines: List[str] = []
    if location_name:
        location_lines.append(f"- 現在の舞台は「{location_name}」。")
    if time_slot or time_str:
        ts = (
            f"{time_slot} / {time_str}"
            if time_slot and time_str
            else (time_slot or time_str)
        )
        location_lines.append(f"- 時間帯は「{ts}」。")

    if others_present_flag is True:
        location_lines.append(
            "- 周囲には他の学院生や利用者がいます。"
            "完全な二人きりではないため、振る舞いは控えめに、甘さはささやかに。"
        )
    elif others_present_flag is False:
        location_lines.append(
            "- 現在、この場には事実上あなたとリセリアだけの二人きりです。"
        )

    affection_label = persona.get_affection_label(affection)

    try:
        guideline = persona.build_emotion_control_guideline(
            affection_with_doki=affection,
            doki_level=doki_level,
            mode_current=mode_current,
        )
    except Exception:
        guideline = ""

    if not guideline:
        guideline = persona._build_default_guideline(
            affection_with_doki=affection,
            doki_level=doki_level,
            mode_current=mode_current,
        )

    masking_note = ""
    if masking_degree >= 0.7:
        masking_note = (
            "※ 現在、表情コントロール（ばけばけ度）が高いため、"
            "内心の恋愛感情や高揚をあえて抑え、"
            "外見上は一段階落ち着いたトーンで振る舞ってください。"
            "特に周囲に他人がいる場合は、あからさまな告白や将来の話は避け、"
            "好意はささやかな言い回しや視線・仕草にとどめてください。"
        )
    elif masking_degree >= 0.3:
        masking_note = (
            "※ 表情コントロール（ばけばけ度）が中程度のため、"
            "強すぎるデレは少し抑えつつ、"
            "さりげない甘さがにじむ程度に留めてください。"
        )

    if masking_env_note:
        masking_note = masking_env_note

    length_guideline = persona._build_length_guideline(length_mode)

    header_lines: List[str] = []
    header_lines.append("[感情・関係性プロファイル]")
    header_lines.append(
        f"- 実効好感度 (affection_with_doki): {affection:.2f} "
        f"(zone={zone}, doki_level={doki_level}, doki_power={doki_power:.1f})"
    )
    if affection_label:
        header_lines.append(f"- 好意の解釈: {affection_label}")

    if relationship_level > 0.0:
        header_lines.append(
            f"- 関係レベル (relationship_level): {relationship_level:.1f} / 100"
        )
        if relationship_stage:
            header_lines.append(f"- 関係ステージ: {relationship_stage}")

    if masking_degree > 0.0:
        header_lines.append(
            f"- 表情コントロール（ばけばけ度）: {masking_degree:.2f} "
            "(0=素直 / 1=完全に平静を装う)"
        )

    header_lines.append(
        f"- 発話の長さモード: {persona._normalize_length_mode(length_mode)} "
        "(short/normal/long/story/auto)"
    )

    if location_lines:
        header_lines.extend(location_lines)

    header_lines.append(
        "- 備考: ドキドキ💓はその場の高揚感、relationship_level は長期的な信頼・絆の指標です。"
    )

    if masking_note:
        header_lines.append(masking_note)

    blocks: List[str] = []
    blocks.append("\n".join(header_lines))

    guideline = (guideline or "").strip()
    if guideline:
        blocks.append(guideline)

    length_guideline = (length_guideline or "").strip()
    if length_guideline:
        blocks.append(length_guideline)

    header_block = "\n\n".join(blocks) + "\n"

    if base_system_prompt:
        new_system_prompt = base_system_prompt.rstrip() + "\n\n" + header_block + "\n"
    else:
        new_system_prompt = header_block + "\n"

    return new_system_prompt
