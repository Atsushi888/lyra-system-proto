# actors/persona/affection_prompt_utils.py
from __future__ import annotations

from typing import Any, Mapping


def _extract_affection_score(emotion: Any) -> float:
    """
    EmotionResult / dict / それ以外、どれが来ても
    0.0〜1.0 の affection スコアを取り出すユーティリティ。
    """
    if emotion is None:
        return 0.0

    # dict っぽいもの
    if isinstance(emotion, Mapping):
        val = (
            emotion.get("affection_with_doki")
            or emotion.get("affection")
            or 0.0
        )
        try:
            return float(val)
        except Exception:
            return 0.0

    # オブジェクト（EmotionResult を想定したダックタイピング）
    try:
        if hasattr(emotion, "affection_with_doki"):
            val = getattr(emotion, "affection_with_doki")
        else:
            val = getattr(emotion, "affection", 0.0)
        return float(val or 0.0)
    except Exception:
        return 0.0


def build_system_prompt_with_affection(
    persona: Any,
    base_system_prompt: str,
    emotion: Any,
    doki_power: float = 0.0,
) -> str:
    """
    Persona + Emotion + doki_power から、
    「好感度レベルに応じたデレ指示入り system_prompt」を組み立てる。

    - persona:
        - リセリアの Persona インスタンス想定だが、
          build_affection_hint_from_score(score: float) を持っていれば他キャラでもよい。
    - base_system_prompt:
        - persona.get_system_prompt() で取得したベース。
    - emotion:
        - EmotionAI / MixerAI 由来の emotion dict または EmotionResult。
    - doki_power:
        - -1.0〜+1.0 程度の補正値（0.0 なら無視）。

    返り値:
        - LLM に渡す最終的な system_prompt。
    """
    # ベースだけは必ず適用
    system_prompt = base_system_prompt or ""

    # affection スコア抽出（0.0〜1.0 にクランプ）
    base_aff = _extract_affection_score(emotion)

    score = base_aff + float(doki_power or 0.0)
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
