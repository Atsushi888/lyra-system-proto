# actors/persona/affection_prompt_utils.py
from __future__ import annotations

from typing import Any, Mapping

from actors.emotion_ai import EmotionResult


def _extract_affection(emotion: Any) -> float:
    """
    EmotionResult または dict / Mapping から affection を安全に取り出すヘルパ。
    それ以外の型の場合は 0.0 を返す。
    """
    if isinstance(emotion, EmotionResult):
        try:
            return float(emotion.affection)
        except Exception:
            return 0.0

    # dict / Mapping 形式の emotion_override も許容
    if isinstance(emotion, Mapping):
        try:
            return float(emotion.get("affection", 0.0) or 0.0)
        except Exception:
            return 0.0

    return 0.0


def _extract_doki_power(doki_power: float | Any, emotion: Any) -> float:
    """
    引数 doki_power を優先しつつ、
    emotion(dict) 側に doki_power が入っていればそれも利用できるようにする。
    """
    # まずは引数を信頼
    try:
        base = float(doki_power or 0.0)
    except Exception:
        base = 0.0

    # emotion が dict なら "doki_power" を上書き候補として見る
    if isinstance(emotion, Mapping):
        try:
            if "doki_power" in emotion:
                base = float(emotion.get("doki_power", base) or base)
        except Exception:
            pass

    # 最終的に 0.0〜1.0 に収める（過剰入力の暴走防止）
    if base < -1.0:
        base = -1.0
    if base > 1.0:
        base = 1.0

    return base


def build_system_prompt_with_affection(
    persona: Any,
    base_system_prompt: str,
    emotion: EmotionResult | Mapping[str, Any] | None,
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
        - EmotionAI.analyze() の結果（EmotionResult）か、
          MixerAI などが吐く dict 形式の emotion_override。
    - doki_power:
        - dokipower_control などから与えられる追加補正。
          （内部で -1.0〜+1.0 にクランプ）

    返り値:
        - LLM に渡す最終的な system_prompt。
    """
    # ベースだけは必ず適用
    system_prompt = base_system_prompt or ""

    if emotion is None:
        return system_prompt

    # 0.0〜1.0 に収まるようざっくりクランプ
    base_aff = _extract_affection(emotion)
    dp = _extract_doki_power(doki_power, emotion)

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
        # ベースの system_prompt の後ろに追記する
        system_prompt = system_prompt.rstrip() + "\n\n" + hint

    return system_prompt
