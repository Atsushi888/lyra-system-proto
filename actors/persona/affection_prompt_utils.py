# actors/persona/affection_prompt_utils.py
from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def _get_effective_emotion_dict(
    llm_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    現在の EmotionResult 相当の dict を取得するヘルパ。

    優先順位:
    1) st.session_state["mixer_debug_emotion"]  … ドキドキ💓調整用（開発／デバッグ）
    2) llm_meta["emotion"]                     … 本番の EmotionAI.analyze() 結果
    """
    emo: Dict[str, Any] = {}

    # 1) Mixer 用デバッグ EmotionResult
    try:
        if hasattr(st, "session_state"):
            val = st.session_state.get("mixer_debug_emotion")
            if isinstance(val, dict):
                emo = val
    except Exception:
        # Streamlit 未初期化などは静かに無視
        emo = {}

    # 2) llm_meta 側（EmotionAI.analyze の結果）
    if not emo and llm_meta:
        val = llm_meta.get("emotion")
        if isinstance(val, dict):
            emo = val

    return emo or {}


def build_system_prompt_with_affection(
    persona: Any,
    base_system_prompt: str,
    llm_meta: Dict[str, Any] | None = None,
) -> str:
    """
    Persona + 現在の EmotionResult 情報から、
    「好感度レベルに応じたデレ指示入り system_prompt」を組み立てる。

    - persona:
        - リセリア Persona インスタンス想定だが、
          persona.build_affection_hint_from_score(score, doki_level=...) を
          実装していれば他キャラでも利用可能。
    - base_system_prompt:
        - もともとのシステムプロンプト（Persona の素の指示）。
    - llm_meta:
        - AnswerTalker が持っている llm_meta 全体。
          ここから emotion dict を取得する（なければ session_state を参照）。

    返り値:
        - LLM に渡す最終的な system_prompt。
    """
    # ベースだけは必ず適用
    system_prompt = base_system_prompt or ""

    # 現在の感情情報を取得
    emo = _get_effective_emotion_dict(llm_meta)
    if not emo:
        return system_prompt

    # affection_with_doki があれば最優先、それが無ければ生の affection を使う
    try:
        score = float(
            emo.get("affection_with_doki", emo.get("affection", 0.0)) or 0.0
        )
    except Exception:
        score = 0.0

    # 0.0〜1.0 にクランプ
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0

    try:
        doki_level = int(emo.get("doki_level", 0) or 0)
    except Exception:
        doki_level = 0

    # Persona 側がヒント生成ヘルパを持っていれば使う
    hint = ""
    if hasattr(persona, "build_affection_hint_from_score"):
        fn = getattr(persona, "build_affection_hint_from_score")
        try:
            # doki_level 引数付きバージョンを優先
            hint = fn(score, doki_level=doki_level)
        except TypeError:
            # 古いシグネチャ（score だけ）の場合
            try:
                hint = fn(score)
            except Exception:
                hint = ""
        except Exception:
            hint = ""

    if hint:
        system_prompt = system_prompt + "\n\n" + str(hint)

    return system_prompt
