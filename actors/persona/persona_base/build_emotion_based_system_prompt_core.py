# actors/persona/persona_base/build_emotion_based_system_prompt_core.py
from __future__ import annotations

from typing import Any, Dict, Optional

from actors.persona.persona_base.build_emotion_header import (
    build_emotion_header_core,
)


def _select_env_suffix(
    persona: Any,
    others_present: Optional[bool],
) -> str:
    """
    world_state.others_present に応じて
    public / private 用の suffix を選択する。
    """
    public_suffix = getattr(persona, "system_prompt_public_suffix", "") or ""
    private_suffix = getattr(persona, "system_prompt_private_suffix", "") or ""

    if others_present is True:
        # 人がいる → public 優先
        return public_suffix or private_suffix
    if others_present is False:
        # 二人きり → private 優先
        return private_suffix or public_suffix
    # 不明 → どちらも未指定なら空
    return public_suffix or private_suffix


def build_emotion_based_system_prompt_core(
    *,
    persona: Any,
    base_system_prompt: str,
    emotion_override: Optional[Dict[str, Any]],
    mode_current: str,
    length_mode: str,
) -> str:
    """
    PersonaBase.build_emotion_based_system_prompt から呼ばれる本体。

    - base_system_prompt は従来 messages 内の system を反映
    - persona.system_prompt_base / *_suffix を優先的に使う
    - emotion_override から world_state / scene_emotion / emotion を見て
      感情ヘッダを構築
    - 最後に文章量ガイドラインを付加
    """
    eo = emotion_override or {}

    world_state = eo.get("world_state") or {}
    if not isinstance(world_state, dict):
        world_state = {}

    scene_emotion = eo.get("scene_emotion") or {}
    if not isinstance(scene_emotion, dict):
        scene_emotion = {}

    emotion_block = eo.get("emotion")

    # others_present 判定
    others_raw = world_state.get("others_present")
    others_present: Optional[bool]
    if isinstance(others_raw, bool):
        others_present = others_raw
    else:
        others_present = None

    # ベースの system_prompt を決定
    base_from_persona = getattr(persona, "system_prompt_base", "") or ""
    core_prompt = base_from_persona or base_system_prompt or ""

    # 環境に応じた suffix
    env_suffix = _select_env_suffix(persona, others_present)

    # 感情ヘッダ
    emotion_header = build_emotion_header_core(
        persona=persona,
        emotion=emotion_block,
        world_state=world_state,
        scene_emotion=scene_emotion,
    )

    # 文章量ガイドライン
    length_guideline = persona._build_length_guideline(length_mode)

    parts = []
    if core_prompt.strip():
        parts.append(core_prompt.strip())
    if env_suffix.strip():
        parts.append(env_suffix.strip())
    if emotion_header.strip():
        parts.append(emotion_header.strip())
    if length_guideline.strip():
        parts.append(length_guideline.strip())

    return "\n\n".join(parts)
