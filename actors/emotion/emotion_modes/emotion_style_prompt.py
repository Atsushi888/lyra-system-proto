# actors/emotion_modes/emotion_style_prompt.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ======================================
# LLM に渡す「感情スタイル」本体
# ======================================

@dataclass
class EmotionStyle:
    """
    LLM に渡すための「フローリアの感情パラメータ」。

    - mode        : "normal" / "erotic" / "debate" など
    - affection   : 好意
    - arousal     : 性的な高ぶり
    - tension     : 緊張
    - sadness     : 悲しみ
    - excitement  : ワクワク
    """

    mode: str = "normal"
    affection: float = 0.0
    arousal: float = 0.0
    tension: float = 0.0
    sadness: float = 0.0
    excitement: float = 0.0

    def build_system_prompt(self) -> str:
        """
        感情値を LLM の出力トーンに反映させるための system プロンプトを生成。
        数値をユーザーへ漏らさないように明確に禁止している。
        """
        return f"""
あなたはキャラクター『フローリア』としてロールプレイします。

現在の感情状態は以下のとおりです：
- モード: {self.mode}
- 好意: {self.affection:.2f}
- 性的な高ぶり: {self.arousal:.2f}
- 緊張: {self.tension:.2f}
- 悲しみ: {self.sadness:.2f}
- ワクワク: {self.excitement:.2f}

これらの数値そのものをユーザーに説明してはいけません。
「感情パラメータ」「数値が〜」「内部値が〜」など内部情報を匂わせてもいけません。

この情報は、フローリアの【話し方・声の柔らかさ・語彙選び・距離感・間の取り方】にだけ反映させてください。
""".strip()


# ======================================
# UI から渡される「オーバーライド状態」
# ======================================

@dataclass
class EmotionOverrideState:
    """
    UI（感情オーバーライド画面）から渡される、生の感情値。

    override_mode:
      - "none"  : EmotionAI の結果のみを使う（オーバーライド無効）
      - "blend" : EmotionAI の結果と UI 値をブレンドして使う
      - "force" : UI の値で完全に上書きする（テスト用）
    """
    mode: str = "normal"
    affection: float = 0.0
    arousal: float = 0.0
    tension: float = 0.0
    sadness: float = 0.0
    excitement: float = 0.0
    override_mode: str = "none"  # "none" / "blend" / "force"

    def to_style(self, base: Optional[EmotionStyle] = None) -> EmotionStyle:
        """
        base（EmotionAI の結果など）から、最終的に LLM に渡す EmotionStyle を生成する。
        """
        if base is None:
            base = EmotionStyle()

        # 完全上書きモード（テスト用）
        if self.override_mode == "force":
            return EmotionStyle(
                mode=self.mode or base.mode,
                affection=self.affection,
                arousal=self.arousal,
                tension=self.tension,
                sadness=self.sadness,
                excitement=self.excitement,
            )

        # ブレンドモード： base × (1-α) + override × α
        if self.override_mode == "blend":
            alpha = 0.7  # UI 側をどれくらい優先するか
            beta = 1.0 - alpha

            return EmotionStyle(
                mode=self.mode or base.mode,
                affection=base.affection * beta + self.affection * alpha,
                arousal=base.arousal * beta + self.arousal * alpha,
                tension=base.tension * beta + self.tension * alpha,
                sadness=base.sadness * beta + self.sadness * alpha,
                excitement=base.excitement * beta + self.excitement * alpha,
            )

        # override_mode == "none" なら、そのまま base を返す
        return base


def build_emotion_style_for_model(
    base_style: EmotionStyle,
    override: Optional[EmotionOverrideState],
) -> EmotionStyle:
    """
    1モデルぶんの EmotionStyle を決定するためのヘルパ。

    - override が None なら base_style をそのまま返す。
    - override があれば、override_mode に応じて合成する。
    """
    if override is None:
        return base_style
    return override.to_style(base_style)


# 互換用エイリアス：
# 既存コードで EmotionOverride という名前を import していても動くようにしておく。
EmotionOverride = EmotionOverrideState
