# actors/persona/build_emotion_based_system_prompt_core.py
from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_emotion_based_system_prompt_core(
    persona: Any,
    *,
    base_system_prompt: str,
    emotion_override: Optional[Dict[str, Any]] = None,
    mode_current: str = "normal",
    length_mode: str = "auto",
) -> str:
    """
    PersonaBase から呼び出されるコア実装。
    DokiPowerControl / world_state_manual_controls から渡された
    world_state["others_present"] を最優先して system_prompt に反映する。
    """

    emotion_override = emotion_override or {}
    world_state = emotion_override.get("world_state") or {}
    emotion = emotion_override.get("emotion") or {}

    # ---------------------------------------------------------
    # ❤️ 感情（affection / doki）
    # ---------------------------------------------------------
    affection = float(
        emotion.get("affection_with_doki", emotion.get("affection", 0.0)) or 0.0
    )
    doki_power = float(emotion.get("doki_power", 0.0) or 0.0)
    doki_level = int(emotion.get("doki_level", 0) or 0)
    zone = str(emotion.get("affection_zone", "auto") or "auto")

    # relationship
    relationship_level = float(
        emotion.get("relationship_level", emotion.get("relationship", 0.0)) or 0.0
    )
    relationship_stage = str(emotion.get("relationship_stage") or "")
    if not relationship_stage and relationship_level > 0.0:
        from actors.emotion.emotion_state import relationship_stage_from_level
        relationship_stage = relationship_stage_from_level(relationship_level)

    # masking
    masking_degree = float(
        emotion.get("masking_degree", emotion.get("masking", 0.0)) or 0.0
    )
    masking_degree = max(0.0, min(1.0, masking_degree))

    # ---------------------------------------------------------
    # 🎭 world_state（場所・時間・周囲の状況）
    # ---------------------------------------------------------
    loc_player = (world_state.get("locations") or {}).get("player")
    location_name = (
        loc_player
        or world_state.get("location_name")
        or world_state.get("player_location")
    )

    time_info = world_state.get("time") or {}
    time_slot = time_info.get("slot") or world_state.get("time_of_day")
    time_str = time_info.get("time_str")

    # ---------------------------------------------------------
    # ⭐️【重要】others_present（DokiPowerControl の最優先フラグ）
    # ---------------------------------------------------------
    # world_state["others_present"] に明示的 bool が来たら一切推定せずそのまま使う
    raw_others_present = world_state.get("others_present", None)
    if isinstance(raw_others_present, bool):
        others_present_flag: Optional[bool] = raw_others_present
    else:
        # 旧ロジック fallback（推定）
        party_mode = (
            world_state.get("party_mode")
            or (world_state.get("party") or {}).get("mode")
        )
        if party_mode in ("others", "group"):
            others_present_flag = True
        elif party_mode in ("alone",):
            others_present_flag = False
        else:
            others_present_flag = None

    is_alone = (others_present_flag is False)

    # ---------------------------------------------------------
    # 🏠 場所による masking 挙動
    # ---------------------------------------------------------
    masking_cfg = persona._get_masking_defaults()
    unmasked_locs = masking_cfg.get("unmasked_locations", [])
    masked_locs = masking_cfg.get("masked_locations", [])
    loc_key = str(location_name or "").lower()

    is_unmasked_place = any(tag in loc_key for tag in unmasked_locs)
    is_masked_place = any(tag in loc_key for tag in masked_locs)

    masking_env_note = ""
    example_line = ""
    raw_example = (masking_cfg.get("rules") or {}).get("example_line")
    if isinstance(raw_example, str) and raw_example.strip():
        example_line = raw_example.replace("{PLAYER_NAME}", persona.player_name)

    if is_unmasked_place:
        masking_env_note = (
            "※ 現在は親しい相手とくつろげる場所にいるため、"
            "表情コントロール（ばけばけ度）はほとんど働かず、自然な甘さがそのまま出て構いません。"
        )
        if example_line:
            masking_env_note += f"\n  例: 「{example_line}」"

    elif is_masked_place:
        if is_alone:
            masking_env_note = (
                "※ 学院内ですが現在は実質二人きりです。"
                "外見上の気取りはそこまで必要なく、素直な恋愛感情を見せても構いません。"
            )
        else:
            masking_env_note = (
                "※ 学院内で人目があるため、ばけばけ度に応じて少し落ち着いた振る舞いを維持してください。"
            )

    # ---------------------------------------------------------
    # 📝 場所説明文
    # ---------------------------------------------------------
    location_lines: List[str] = []
    if location_name:
        location_lines.append(f"- 現在の舞台は「{location_name}」。")

    if time_slot or time_str:
        ts = (
            f"{time_slot} / {time_str}"
            if time_slot and time_str else (time_slot or time_str)
        )
        location_lines.append(f"- 時間帯は「{ts}」。")

    # ---------------------------------------------------------
    # 👥 周囲に人がいるか（system_prompt へ明示的に書く）
    # ---------------------------------------------------------
    if others_present_flag is True:
        location_lines.append(
            "- 周囲には他の学院生や利用者がいます。"
            "完全な二人きりではないため、振る舞いは控えめに、甘さはささやかに。"
        )
    elif others_present_flag is False:
        location_lines.append(
            "- 現在、この場には事実上あなたとリセリアだけの二人きりです。"
        )

    # ---------------------------------------------------------
    # ❤️ 好意ラベル
    # ---------------------------------------------------------
    affection_label = persona.get_affection_label(affection)

    # ---------------------------------------------------------
    # 🎛️ ガイドライン
    # ---------------------------------------------------------
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

    # masking 注意書き
    masking_note = ""
    if masking_env_note:
        masking_note = masking_env_note
    else:
        if masking_degree >= 0.7:
            masking_note = (
                "※ 現在、表情コントロールが高いため、外見上は落ち着いた振る舞いを保ちつつ、"
                "内心の甘さは仕草やささやかな一言ににじませてください。"
            )
        elif masking_degree >= 0.3:
            masking_note = (
                "※ 表情コントロールが中程度のため、強すぎる甘さは少し抑えつつ、"
                "自然な柔らかさが伝わる範囲での表現が望ましいです。"
            )

    length_guideline = persona._build_length_guideline(length_mode)

    # ---------------------------------------------------------
    # 🧩 ヘッダ組み立て
    # ---------------------------------------------------------
    header_lines: List[str] = []
    header_lines.append("[感情・関係性プロファイル]")
    header_lines.append(
        f"- 実効好感度: {affection:.2f} (zone={zone}, doki_level={doki_level}, doki_power={doki_power:.1f})"
    )
    if affection_label:
        header_lines.append(f"- 好意の解釈: {affection_label}")

    if relationship_level > 0.0:
        header_lines.append(
            f"- 関係レベル: {relationship_level:.1f} / 100"
        )
        if relationship_stage:
            header_lines.append(f"- 関係ステージ: {relationship_stage}")

    header_lines.append(
        f"- 表情コントロール（ばけばけ度）: {masking_degree:.2f} "
        "(0=素直 / 1=完全に平静を装う)"
    )

    header_lines.append(
        f"- 発話の長さモード: {persona._normalize_length_mode(length_mode)}"
    )

    if location_lines:
        header_lines.extend(location_lines)

    header_lines.append(
        "- 備考: ドキドキ💓は短期刺激、relationship_level は長期的な信頼の指標です。"
    )

    if masking_note:
        header_lines.append(masking_note)

    # ---------------------------------------------------------
    # 🧱 最終連結
    # ---------------------------------------------------------
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
