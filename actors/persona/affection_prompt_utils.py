# actors/persona/affection_prompt_utils.py
from __future__ import annotations

from typing import Any

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
        - persona.get_system_prompt() で取得したベース、もしくは
          PersonaAI などから組み立てた通常の system プロンプト。
    - emotion:
        - EmotionAI.analyze() の結果、または MixerAI が組んだ EmotionResult 相当。
    - doki_power:
        - dokipower_control などから与えられる追加補正（0〜100想定）。
          ここでは 0〜1.0 に正規化して affection に足し込む。
    """
    system_prompt = base_system_prompt or ""

    # 感情情報がなければベースだけ返す
    if emotion is None:
        return system_prompt

    # もともとの affection を取得
    base_aff = float(getattr(emotion, "affection", 0.0) or 0.0)

    # ドキドキ💓補正（0〜100 → 0〜1.0 に正規化して弱めに効かせる）
    try:
        dp_raw = float(doki_power)
    except Exception:
        dp_raw = 0.0

    # 100 で +0.3 くらいに抑える（好感度 1.0 を踏み越えすぎないように）
    dp = max(0.0, min(dp_raw, 100.0)) / 100.0 * 0.3

    # affection_with_doki が EmotionResult 側で計算されているならそれを優先
    if hasattr(emotion, "affection_with_doki"):
        try:
            score = float(getattr(emotion, "affection_with_doki"))
        except Exception:
            score = base_aff + dp
    else:
        score = base_aff + dp

    # 0.0〜1.0 にクランプ
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
        # 元の system_prompt の末尾に、空行を挟んで追記
        system_prompt = system_prompt.rstrip() + "\n\n" + hint.strip()

    return system_prompt
