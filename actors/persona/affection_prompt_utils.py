# actors/persona/affection_prompt_utils.py
from __future__ import annotations

from typing import Any

from actors.persona.persona_classes.persona_riseria_ja import Persona as RiseriaPersona
from actors.emotion_ai import EmotionResult


def build_system_prompt_with_affection(
    persona: Any,
    base_system_prompt: str,
    emotion: EmotionResult | None,
    doki_power: float = 0.0,
) -> str:
    """
    Persona + EmotionResult + doki_power から、
    「好感度レベルに応じたデレ指示入り system_prompt」を組み立てる。

    - persona:
        - リセリアの Persona インスタンス想定だが、
          build_affection_hint_from_score(score: float) を持っていれば他キャラでもよい。
    - base_system_prompt:
        - persona.get_system_prompt() で取得したベース。
    - emotion:
        - EmotionAI.analyze() の結果（直近ターンの EmotionResult）。
    - doki_power:
        - dokipower_control などから与えられる追加補正（-1.0〜+1.0 程度を想定）。

    返り値:
        - LLM に渡す最終的な system_prompt。
    """
    # ベースだけは必ず適用
    system_prompt = base_system_prompt or ""

    if emotion is None:
        return system_prompt

    # 0.0〜1.0 に収まるようざっくりクランプ
    base_aff = float(getattr(emotion, "affection", 0.0) or 0.0)
    try:
        dp = float(doki_power)
    except Exception:
        dp = 0.0

    score = base_aff + dp
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0

    # Persona 側がヒント生成ヘルパを持っていれば使う
    hint = ""
    if hasattr(persona, "build_affection_hint_from_score"):
        try:
            hint = persona.build_affection_hint_from_score(score)
        except Exception:
            hint = ""

    if hint:
        system_prompt = system_prompt + "\n\n" + hint

    return system_prompt
