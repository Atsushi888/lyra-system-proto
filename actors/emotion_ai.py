# actors/emotion_ai.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List
import json

from llm.llm_manager import LLMManager


@dataclass
class EmotionResult:
    """
    フローリア自身の「表向きの感情状態」を表すスナップショット。
    """
    mode: str = "normal"        # "normal" / "erotic" / "debate" など
    affection: float = 0.0      # 好意・親しみ
    arousal: float = 0.0        # 性的な高ぶり
    tension: float = 0.0        # 緊張・不安
    anger: float = 0.0
    sadness: float = 0.0
    excitement: float = 0.0
    raw_text: str = ""          # LLM の生返答（JSONそのもの or エラー）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EmotionAI:
    """
    ComposerAI の最終返答と MemoryAI の記憶コンテキストから、
    フローリアの感情状態を推定するクラス。

    - LLMManager を使って 1 モデル（推奨: gpt-5.1）を叩く
    - JSON で感情値 + mode を返させる
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        model_name: str = "gpt51",
    ) -> None:
        self.llm_manager = llm_manager
        self.model_name = model_name

    # ---------------------------------------------
    def _build_messages(
        self,
        composer: Dict[str, Any],
        memory_context: str = "",
        user_text: str = "",
    ) -> List[Dict[str, str]]:
        """
        OpenAI 互換の messages 配列を組み立てる。
        """

        final_text = (composer.get("text") or "").strip()
        source_model = composer.get("source_model", "")

        system_prompt = """
あなたは「感情解析専用 AI」です。
以下の情報から、キャラクター『フローリア』本人の感情状態を推定してください。

必ず **JSON オブジェクトのみ** を出力してください。
説明文やコメント、日本語の文章などは一切書かないでください。

JSON 形式は以下です：

{
  "mode": "normal" | "erotic" | "debate",
  "affection": 0.0〜1.0,
  "arousal": 0.0〜1.0,
  "tension": 0.0〜1.0,
  "anger": 0.0〜1.0,
  "sadness": 0.0〜1.0,
  "excitement": 0.0〜1.0
}

- affection: プレイヤーへの親しみ・信頼・好意
- arousal: ロマンチック／性的な高ぶり
- tension: 緊張・不安・張り詰めた感覚
- anger: 怒り・苛立ち
- sadness: 悲しみ・寂しさ
- excitement: ワクワク・高揚感

mode の決め方（目安）：
  - "erotic" : arousal が 0.6 以上かつ affection も 0.5 以上の場合
  - "debate" : tension または anger が 0.6 以上の場合
  - "normal" : 上記に当てはまらない場合全て
"""

        desc_lines: List[str] = []

        desc_lines.append("=== Latest composed reply of Floria ===")
        if source_model:
            desc_lines.append(f"(source_model: {source_model})")
        desc_lines.append(final_text or "(empty)")

        if user_text:
            desc_lines.append("\n=== Latest user text ===")
            desc_lines.append(user_text)

        if memory_context:
            desc_lines.append("\n=== Memory context (long-term / background) ===")
            desc_lines.append(memory_context)

        user_prompt = "\n".join(desc_lines)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    # ---------------------------------------------
    def analyze(
        self,
        composer: Dict[str, Any],
        memory_context: str = "",
        user_text: str = "",
    ) -> EmotionResult:
        """
        感情推定のメインメソッド。

        - composer: AnswerTalker.llm_meta["composer"]
        - memory_context: AnswerTalker.llm_meta["memory_context"]
        - user_text: 直近のユーザー入力
        """
        messages = self._build_messages(
            composer=composer,
            memory_context=memory_context,
            user_text=user_text or "",
        )

        try:
            raw = self.llm_manager.call_model(
                model_name=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=320,
            )

            # call_model の返り値が (text, usage) or str 想定
            if isinstance(raw, tuple):
                text = raw[0]
            else:
                text = raw

            if not isinstance(text, str):
                text = "" if text is None else str(text)

            data = json.loads(text)

            res = EmotionResult(
                mode=str(data.get("mode", "normal")),
                affection=float(data.get("affection", 0.0)),
                arousal=float(data.get("arousal", 0.0)),
                tension=float(data.get("tension", 0.0)),
                anger=float(data.get("anger", 0.0)),
                sadness=float(data.get("sadness", 0.0)),
                excitement=float(data.get("excitement", 0.0)),
                raw_text=text,
            )
            return res

        except Exception as e:
            # 解析失敗時でもシステム全体は止めない
            return EmotionResult(
                mode="normal",
                raw_text=f"[EmotionAI error] {e}",
            )
