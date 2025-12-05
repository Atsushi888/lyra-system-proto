# actors/persona/affection_prompt_utils.py
from __future__ import annotations

from typing import Any, Optional

from actors.emotion_ai import EmotionResult


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def build_system_prompt_with_affection(
    persona: Any,
    base_system_prompt: str,
    emotion: Optional[EmotionResult],
    doki_power: float = 0.0,
) -> str:
    """
    Persona + EmotionResult + doki_power から、
    「好感度レベルに応じたデレ指示入り system_prompt」を組み立てる。

    - persona:
        - リセリアの Persona インスタンス想定だが、
          `build_affection_hint_from_score(score: float)` を
          実装していれば他キャラでも利用可能。
    - base_system_prompt:
        - persona 由来の素の system_prompt。
    - emotion:
        - EmotionAI.analyze() の結果、または dokipower デバッグから再構成した EmotionResult。
    - doki_power:
        - dokipower_control などから与えられる追加補正。
          （0〜100 を想定し、ここでは 0〜1 に正規化して加算）

    返り値:
        - LLM に渡す最終的な system_prompt 文字列。
    """
    system_prompt = base_system_prompt or ""

    if emotion is None:
        # 感情情報がなければベースだけ返す
        return system_prompt

    # 0. ベース好感度
    base_aff = float(getattr(emotion, "affection", 0.0) or 0.0)

    # 1. ドキドキ💓パワーを 0.0〜1.0 にざっくり正規化して加算
    try:
        dp_raw = float(doki_power)
    except Exception:
        dp_raw = 0.0

    # 0〜100 想定で 100 → +0.5 くらいのゲタをイメージ
    dp = (dp_raw / 100.0) * 0.5
    score = _clamp01(base_aff + dp)

    # 2. Persona 側がヒント生成ヘルパを持っていれば使う
    hint = ""
    if hasattr(persona, "build_affection_hint_from_score"):
        try:
            hint = persona.build_affection_hint_from_score(score)
        except Exception:
            hint = ""

    # 3. ヒントがあれば末尾に追記
    if hint:
        system_prompt = system_prompt.rstrip() + "\n\n" + hint.strip()

    return system_prompt
