# actors/emotion_modes/emotion_style_prompt.py
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class EmotionStyle:
    """
    LLM に渡すための「フローリアの感情パラメータ」。
    - mode        : "normal" / "erotic" / "debate"
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
