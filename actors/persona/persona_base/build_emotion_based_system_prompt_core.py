# actors/persona/persona_base/build_emotion_based_system_prompt_core.py
from __future__ import annotations

from typing import Any, Dict, Optional
import os

import streamlit as st  # デバッグ用

from actors.persona.persona_base.persona_base import PersonaBase
from actors.utils.debug_world_state import debug_world_state  # 共通デバッガ

LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"


def _select_relationship_stage(level: float) -> str:
    """
    relationship_level (0-100) → ざっくりステージ名。
    DokiPowerController の解釈と揃え気味にしておく。
    """
    if level >= 80:
        return "soulmate"
    if level >= 60:
        return "dating"
    if level >= 40:
        return "close_friends"
    if level >= 20:
        return "friendly"
    return "acquaintance"


def _build_environment_summary(
    *,
    persona: PersonaBase,
    world_state: Dict[str, Any],
) -> str:
    """world_state から現在のシーン情報を 1 ブロックのテキストにまとめる。"""

    loc = world_state.get("locations") or {}
    if not isinstance(loc, dict):
        loc = {}
    location = loc.get("player") or loc.get("floria") or "プレイヤーの部屋"

    t = world_state.get("time") or {}
    if not isinstance(t, dict):
        t = {}
    slot = t.get("slot", "morning")
    time_str = t.get("time_str", "07:30")

    weather = world_state.get("weather", "clear")
    party = world_state.get("party") or {}
    if not isinstance(party, dict):
        party = {}
    party_mode = party.get("mode", "both")

    others_present_raw = world_state.get("others_present")
    others_present = bool(others_present_raw) if isinstance(others_present_raw, bool) else None

    lines: list[str] = []
    lines.append(f"- 現在の舞台は「{location}」。")
    lines.append(f"- 時間帯は「{slot} / {time_str}」。")

    # others_present があれば、ここで環境ニュアンスを明示
    if others_present is True:
        lines.append(
            "- 周囲には他の学院生や利用者がいます。完全な二人きりではないため、"
            "振る舞いは控えめに、甘さはささやかに。"
        )
    elif others_present is False:
        lines.append(
            "- 現在、この場には事実上あなたと"
            f"{persona.display_name}だけの二人きりです。"
        )

    # 天気と party_mode は必要に応じて
    lines.append(f"- 天候: {weather} / party_mode: {party_mode}。")

    return "\n".join(lines)


def _build_masking_note(
    *,
    persona: PersonaBase,
    masking_degree: float,
    world_state: Dict[str, Any],
) -> str:
    """
    masking_degree と masking_defaults をもとに、
    「どの程度デレを抑えるか」の注釈を返す。
    """
    md = max(0.0, min(masking_degree, 1.0))
    defaults = persona._get_masking_defaults()
    default_level = float(defaults.get("default_level", 0.0) or 0.0)

    # ゆるめの日本語メッセージだけ付けておく
    if md < 0.2:
        level_msg = "ほぼ感情ダダ漏れ状態。素直な喜びや照れがそのまま表情や言い回しに出ても構いません。"
    elif md < 0.4:
        level_msg = "やや感情が表に出やすい状態。基本は素直だが、あまりに露骨なデレだけ少し抑える程度に。"
    elif md < 0.7:
        level_msg = (
            "ある程度は感情を隠せる状態。強すぎるデレや露骨な好意表現は一歩引き、"
            "さりげない視線や言葉選びで好意をにじませてください。"
        )
    else:
        level_msg = (
            "かなり表情をコントロールできる状態。よほどのことがない限り、"
            "表面上は落ち着いたトーンを保ち、内心はモノローグやわずかな描写に留めます。"
        )

    lines: list[str] = []
    lines.append(
        f"- 表情コントロール（ばけばけ度）: {md:.2f} "
        "(0=素直 / 1=完全に平静を装う)"
    )
    lines.append(f"※ {level_msg}")

    if default_level > 0:
        lines.append(
            f"（参考: persona のデフォルトばけばけ度は {default_level:.2f} です）"
        )

    return "\n".join(lines)


def build_emotion_based_system_prompt_core(
    *,
    persona: PersonaBase,
    base_system_prompt: str,
    emotion_override: Optional[Dict[str, Any]],
    mode_current: str,
    length_mode: str,
) -> str:
    """
    PersonaBase.build_emotion_based_system_prompt から呼ばれる本体。

    - world_state.others_present を見て public / private の suffix を選択
    - 感情・関係性プロファイルをヘッダとして付与
    - reply_length_mode に応じた文章量ガイドラインも末尾に付ける
    """

    emotion_override = emotion_override or {}
    world_state: Dict[str, Any] = emotion_override.get("world_state") or {}
    if not isinstance(world_state, dict):
        world_state = {}
    scene_emotion: Dict[str, Any] = emotion_override.get("scene_emotion") or {}
    if not isinstance(scene_emotion, dict):
        scene_emotion = {}
    emotion_block: Dict[str, Any] = emotion_override.get("emotion") or {}
    if not isinstance(emotion_block, dict):
        emotion_block = {}

    # ===== 1) 環境別 system_prompt の土台を決定 =====
    others_present_raw = world_state.get("others_present")
    others_present: Optional[bool] = None
    if isinstance(others_present_raw, bool):
        others_present = others_present_raw

    # まず共通ベース
    if persona.system_prompt_base:
        system_prompt_env = persona.system_prompt_base
    else:
        system_prompt_env = base_system_prompt

    # public / private suffix を world_state に応じて追加
    if others_present is True and persona.system_prompt_public_suffix:
        system_prompt_env = (
            system_prompt_env.rstrip()
            + "\n\n"
            + persona.system_prompt_public_suffix.strip()
        )
    elif others_present is False and persona.system_prompt_private_suffix:
        system_prompt_env = (
            system_prompt_env.rstrip()
            + "\n\n"
            + persona.system_prompt_private_suffix.strip()
        )

    # ===== 2) 感情・関係性プロファイルを構築 =====
    affection = float(emotion_block.get("affection", 0.0) or 0.0)
    doki_power = float(emotion_block.get("doki_power", 0.0) or 0.0)
    doki_level = int(emotion_block.get("doki_level", 0) or 0)
    relationship_level = float(emotion_block.get("relationship_level", 0.0) or 0.0)
    masking_degree = float(emotion_block.get("masking_degree", 0.0) or 0.0)

    # 現状はシンプルに「affection_with_doki = affection」
    affection_with_doki = affection

    # 好意ラベル（JSON から取れればそれを使う）
    aff_label = persona.get_affection_label(affection_with_doki)
    if not aff_label:
        # フォールバック（少しざっくり）
        if affection_with_doki < 0.15:
            aff_label = "ほぼ他人に近い。まだ強い好意は芽生えていない。"
        elif affection_with_doki < 0.4:
            aff_label = "尊敬と好感がじわじわ育っている段階の相手。"
        elif affection_with_doki < 0.7:
            aff_label = "かなり信頼し、強い好意を自覚し始めている。"
        else:
            aff_label = "深く愛しており、人生レベルで大切な存在として見ている。"

    rel_stage = _select_relationship_stage(relationship_level)

    # ===== 3) ヘッダテキスト組み立て =====
    header_lines: list[str] = []
    header_lines.append("[感情・関係性プロファイル]")
    header_lines.append(
        f"- 実効好感度 (affection_with_doki): "
        f"{affection_with_doki:.2f} (zone=auto, doki_level={doki_level}, doki_power={doki_power:.1f})"
    )
    header_lines.append(f"- 好意の解釈: {aff_label}")
    header_lines.append(
        f"- 関係レベル (relationship_level): {relationship_level:.1f} / 100"
    )
    header_lines.append(f"- 関係ステージ: {rel_stage}")
    header_lines.append(
        f"- 表情コントロール（ばけばけ度）: {masking_degree:.2f} "
        "(0=素直 / 1=完全に平静を装う)"
    )
    header_lines.append(
        f"- 発話の長さモード: {length_mode} (short/normal/long/story/auto)"
    )

    # シーン情報
    header_lines.append(_build_environment_summary(
        persona=persona,
        world_state=world_state,
    ))

    header_lines.append(
        "- 備考: ドキドキ💓はその場の高揚感、relationship_level は長期的な信頼・絆の指標です。"
    )

    # マスキング注釈
    header_lines.append(
        _build_masking_note(
            persona=persona,
            masking_degree=masking_degree,
            world_state=world_state,
        )
    )

    # doki / mode に応じた口調ガイドライン（JSON or デフォルト）
    guideline = persona.build_emotion_control_guideline(
        affection_with_doki=affection_with_doki,
        doki_level=doki_level,
        mode_current=mode_current,
    )
    header_lines.append("")
    header_lines.append(guideline)

    # 文章量ガイドライン
    length_guideline = persona._build_length_guideline(length_mode)
    if length_guideline:
        header_lines.append("")
        header_lines.append(length_guideline)

    emotion_header_text = "\n".join(header_lines)

    # ===== 4) デバッグ出力 =====
    try:
        debug_world_state(
            caller="build_emotion_based_system_prompt_core",
            world_state=world_state,
            scene_emotion=scene_emotion,
            emotion=emotion_block,
            extra={
                "relation_level": relationship_level,
                "masking_degree": masking_degree,
                "length_mode": length_mode,
                "mode_current": mode_current,
            },
        )
    except Exception as e:
        if LYRA_DEBUG:
            st.write(
                "[LYRA DEBUG] PromptCore debug_world_state error:",
                str(e),
            )

    if LYRA_DEBUG:
        st.write("==== [LYRA DEBUG] PromptCore from build_emotion_based_system_prompt_core ===")
        st.json(
            {
                "system_prompt_env_preview": system_prompt_env[:200],
                "world_state": world_state,
                "scene_emotion": scene_emotion,
                "emotion": emotion_block,
            }
        )

    # ===== 5) 最終 system_prompt を返す =====
    final_parts = [system_prompt_env.rstrip(), "", emotion_header_text]
    return "\n".join(final_parts).rstrip()
