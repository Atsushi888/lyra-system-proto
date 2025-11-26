# actors/llm_adapters/emotion_style_prompt.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EmotionStyleHint:
    """
    LLM に渡す「感情スタイル」の簡易コンテナ。

    - mode:       "normal" / "erotic" / "debate" など
    - affection:  好意・親しみ
    - arousal:    性的な高ぶり
    - tension:    緊張
    - sadness:    悲しみ
    - excitement: ワクワク・高揚
    """
    mode: str = "normal"
    affection: float = 0.0
    arousal: float = 0.0
    tension: float = 0.0
    sadness: float = 0.0
    excitement: float = 0.0

    @classmethod
    def from_any(cls, src: Any) -> "EmotionStyleHint":
        """
        EmotionResult / JudgeSignal / dict など、
        それっぽいオブジェクトから安全に取り出して変換する。
        """
        if src is None:
            return cls()

        if isinstance(src, cls):
            return src

        data: Dict[str, Any] = {}
        if isinstance(src, dict):
            data = src
        else:
            # 属性として持っている場合（EmotionResult / JudgeSignal など）
            for key in ("mode", "affection", "arousal",
                        "tension", "sadness", "excitement"):
                if hasattr(src, key):
                    data[key] = getattr(src, key)

        return cls(
            mode=str(data.get("mode", "normal") or "normal"),
            affection=float(data.get("affection", 0.0) or 0.0),
            arousal=float(data.get("arousal", 0.0) or 0.0),
            tension=float(data.get("tension", 0.0) or 0.0),
            sadness=float(data.get("sadness", 0.0) or 0.0),
            excitement=float(data.get("excitement", 0.0) or 0.0),
        )


def build_emotion_style_system_prompt(hint: EmotionStyleHint) -> str:
    """
    実際に LLM に渡す system プロンプト文を構築する。

    ※ 数値そのものを喋らせないよう、メタ説明は禁止する指示を含める。
    """
    return f"""あなたはキャラクター『フローリア』としてロールプレイします。

現在の感情状態は次のとおりです：
- モード: {hint.mode}
- 好意: {hint.affection:.2f}
- 性的な高ぶり: {hint.arousal:.2f}
- 緊張: {hint.tension:.2f}
- 悲しみ: {hint.sadness:.2f}
- ワクワク: {hint.excitement:.2f}

これらの数値そのものを説明したり、数値を引用したりしてはいけません。
フローリアの台詞や地の文のトーン・言葉選び・間の取り方・比喩にさりげなく反映させてください。
数値や「感情スコア」という言葉についてのメタな説明は一切行わず、
あくまで自然な一人称の会話として表現してください。
"""


def inject_emotion_style_system_prompt(
    messages: List[Dict[str, str]],
    hint_source: Any,
    extra_system: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    既存 messages に「感情スタイル system プロンプト」をいい感じにマージする。

    - すでに先頭に system がある場合 → content を結合して 1 個にまとめる
    - system が無い場合           → 先頭に 1 個追加
    - extra_system が渡された場合 → 感情プロンプトの後ろに追記する
    """
    hint = EmotionStyleHint.from_any(hint_source)
    emo_sys = build_emotion_style_system_prompt(hint)

    base_messages = list(messages) if messages is not None else []
    merged_system = emo_sys

    if extra_system:
        merged_system = emo_sys + "\n\n" + str(extra_system)

    if base_messages and base_messages[0].get("role") == "system":
        # 既存 system に前置きする
        old = base_messages[0].get("content", "")
        new_first = {
            "role": "system",
            "content": merged_system + "\n\n" + str(old),
        }
        return [new_first] + base_messages[1:]

    # system が無い場合は先頭に新設
    return [{"role": "system", "content": merged_system}] + base_messages
