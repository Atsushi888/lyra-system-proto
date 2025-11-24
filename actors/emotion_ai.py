# actors/emotion_ai.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
import json

from llm.llm_manager import LLMManager


# ==============================
# データ構造（短期・長期）
# ==============================

@dataclass
class EmotionResult:
    """
    フローリア自身の「短期的な感情状態」を表すスナップショット。
    （1ターンごと / ComposerAI の最終返答ベース）
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


@dataclass
class RelationEmotion:
    """
    特定の相手（旅人、黒騎士、師匠など）に対する
    長期的な感情パラメータ。

    値の範囲は 0.0〜1.0 を推奨。
    """
    affection: float = 0.0    # 好意・愛情
    trust: float = 0.0        # 信頼
    anger: float = 0.0        # 怒り・憎悪
    fear: float = 0.0         # 恐怖
    sadness: float = 0.0      # 悲しみ
    jealousy: float = 0.0     # 嫉妬
    attraction: float = 0.0   # 性的/ロマンチックな惹かれ


@dataclass
class LongTermEmotion:
    """
    キャラ全体の「長期的な感情状態」。

    MemoryAI が提供する長期記憶（イベント）を元に、
    EmotionAI が解析して蓄積する。
    """
    # 世界全体に対するムード
    # 例: {"hope": 0.6, "loneliness": 0.3, "despair": 0.1}
    global_mood: Dict[str, float] = field(default_factory=dict)

    # 各キャラごとの感情
    # 例:
    #   "traveler":    RelationEmotion(affection=0.8, trust=0.7, ...)
    #   "black_knight": RelationEmotion(anger=0.9, fear=0.4, ...)
    relations: Dict[str, RelationEmotion] = field(default_factory=dict)

    # 最終更新されたラウンド数
    last_updated_round: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_mood": dict(self.global_mood),
            "relations": {
                name: asdict(emotion) for name, emotion in self.relations.items()
            },
            "last_updated_round": self.last_updated_round,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LongTermEmotion":
        lt = cls()
        lt.global_mood = data.get("global_mood", {}) or {}
        lt.last_updated_round = int(data.get("last_updated_round", 0) or 0)

        relations_raw = data.get("relations", {}) or {}
        for name, emo in relations_raw.items():
            if isinstance(emo, dict):
                lt.relations[name] = RelationEmotion(**emo)
        return lt


# ==============================
# EmotionAI 本体
# ==============================

class EmotionAI:
    """
    ComposerAI の最終返答と MemoryAI の記憶コンテキスト・記憶レコードから、
    フローリアの感情状態（短期・長期）を推定するクラス。

    - analyze():  短期的な感情（EmotionResult）を推定
    - update_long_term():  MemoryRecord 群から長期感情 LongTermEmotion を更新
    - decide_judge_mode(): 短期＋長期の感情から judge_mode を決定
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        model_name: str = "gpt51",
    ) -> None:
        self.llm_manager = llm_manager
        self.model_name = model_name

        # 長期的な感情状態（キャッシュ）
        self.long_term: LongTermEmotion = LongTermEmotion()

        # 直近ターンの短期感情（analyze() の結果）
        self.last_short_result: Optional[EmotionResult] = None

    # ---------------------------------------------
    # 短期感情：Composer + memory_context ベース
    # ---------------------------------------------
    def _build_messages(
        self,
        composer: Dict[str, Any],
        memory_context: str = "",
        user_text: str = "",
    ) -> List[Dict[str, str]]:
        """
        OpenAI 互換の messages 配列を組み立てる。
        （短期感情推定用）
        """

        final_text = (composer.get("text") or "").strip()
        source_model = composer.get("source_model", "")

        system_prompt = """
あなたは「感情解析専用 AI」です。
以下の情報から、キャラクター『フローリア』本人の短期的な感情状態を推定してください。

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

    def analyze(
        self,
        composer: Dict[str, Any],
        memory_context: str = "",
        user_text: str = "",
    ) -> EmotionResult:
        """
        短期的な感情スコアを推定するメイン関数。

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
            self.last_short_result = res
            return res

        except Exception as e:
            # 解析失敗時でもシステム全体は止めない
            err = EmotionResult(
                mode="normal",
                raw_text=f"[EmotionAI short-term error] {e}",
            )
            self.last_short_result = err
            return err

    # ---------------------------------------------
    # 長期感情：MemoryRecord 群ベース
    # ---------------------------------------------
    def _build_long_term_messages(
        self,
        memory_records: List[Any],
    ) -> List[Dict[str, str]]:
        """
        LongTermEmotion 更新用のプロンプトを作成。

        memory_records は MemoryAI.get_all_records() 等で取得した
        MemoryRecord のリストを想定するが、
        dict でも attr アクセスでも対応できるようにしている。
        """

        system_prompt = """
あなたは「長期感情解析専用 AI」です。

以下に、キャラクター『フローリア』に関する重要な記憶の一覧があります。
それぞれの記憶は、フローリアの人生や対人関係に影響を与えています。

仕事は以下です：

1. 記憶から「フローリアが世界全体をどう感じているか」を推定し、
   global_mood として 0.0〜1.0 の数値で表してください。
   例: hope, loneliness, despair, calmness など（ラベルは自由ですが英単語で）

2. 記憶から「フローリアが特定の人物に対してどのような感情を抱いているか」を推定し、
   relations として 0.0〜1.0 の数値で表してください。
   - キーは短い英語ID（例: "traveler", "black_knight", "priestess" など）にしてください。
   - 各人物について、以下のフィールドを出力してください：
       affection, trust, anger, fear, sadness, jealousy, attraction

必ず **次の形式の JSON オブジェクトのみ** を出力してください。
説明文、日本語、コメントは一切書かないでください。

{
  "global_mood": {
    "hope": 0.7,
    "loneliness": 0.2,
    "despair": 0.1
  },
  "relations": {
    "traveler": {
      "affection": 0.9,
      "trust": 0.85,
      "anger": 0.0,
      "fear": 0.0,
      "sadness": 0.2,
      "jealousy": 0.1,
      "attraction": 0.95
    },
    "black_knight": {
      "affection": 0.0,
      "trust": 0.0,
      "anger": 0.95,
      "fear": 0.4,
      "sadness": 0.3,
      "jealousy": 0.0,
      "attraction": 0.0
    }
  }
}
"""

        lines: List[str] = []
        lines.append("=== Important memories of Floria ===")

        # ざっくりと各レコードを1〜2行でテキスト化
        for idx, rec in enumerate(memory_records, start=1):
            # MemoryRecord の可能性
            summary = getattr(rec, "summary", None)
            importance = getattr(rec, "importance", None)
            tags = getattr(rec, "tags", None)
            round_id = getattr(rec, "round_id", None)

            # dict の可能性
            if summary is None and isinstance(rec, dict):
                summary = rec.get("summary")
                importance = rec.get("importance")
                tags = rec.get("tags")
                round_id = rec.get("round_id")

            summary = summary or ""
            imp = importance if importance is not None else "?"
            tag_str = ", ".join(tags) if tags else ""
            rid = round_id if round_id is not None else "?"

            lines.append(f"- #{idx} (round={rid}, importance={imp}, tags=[{tag_str}]): {summary}")

        user_prompt = "\n".join(lines)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _smooth(self, old: float, new: float, alpha: float = 0.3) -> float:
        """
        長期感情をなめらかに変化させるための簡易スムージング関数。
        """
        if old is None:
            return new
        return float(old) * (1.0 - alpha) + float(new) * alpha

    def update_long_term(
        self,
        memory_records: List[Any],
        current_round: int = 0,
        alpha: float = 0.3,
    ) -> LongTermEmotion:
        """
        MemoryRecord のリストから LongTermEmotion を更新する。

        - memory_records: MemoryAI から渡される重要な記憶一覧
        - current_round:  今のラウンド番号（0 でも可）
        - alpha:           スムージング係数（0.0〜1.0、デフォルトは 0.3）

        戻り値:
            self.long_term （更新後の状態）
        """
        if not memory_records:
            # 記憶が無ければ何も更新しない
            return self.long_term

        messages = self._build_long_term_messages(memory_records)

        try:
            raw = self.llm_manager.call_model(
                model_name=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
            )

            if isinstance(raw, tuple):
                text = raw[0]
            else:
                text = raw

            if not isinstance(text, str):
                text = "" if text is None else str(text)

            data = json.loads(text)
            parsed = LongTermEmotion.from_dict(data)

            # 既存 long_term とスムージングしながらマージ
            merged = LongTermEmotion()
            merged.last_updated_round = current_round or self.long_term.last_updated_round

            # global_mood
            all_mood_keys = set(self.long_term.global_mood.keys()) | set(parsed.global_mood.keys())
            for key in all_mood_keys:
                old_val = self.long_term.global_mood.get(key, 0.0)
                new_val = parsed.global_mood.get(key, 0.0)
                merged.global_mood[key] = self._smooth(old_val, new_val, alpha=alpha)

            # relations
            all_rel_keys = set(self.long_term.relations.keys()) | set(parsed.relations.keys())
            for rel_name in all_rel_keys:
                old_rel = self.long_term.relations.get(rel_name, RelationEmotion())
                new_rel = parsed.relations.get(rel_name, RelationEmotion())

                merged.relations[rel_name] = RelationEmotion(
                    affection=self._smooth(old_rel.affection, new_rel.affection, alpha=alpha),
                    trust=self._smooth(old_rel.trust, new_rel.trust, alpha=alpha),
                    anger=self._smooth(old_rel.anger, new_rel.anger, alpha=alpha),
                    fear=self._smooth(old_rel.fear, new_rel.fear, alpha=alpha),
                    sadness=self._smooth(old_rel.sadness, new_rel.sadness, alpha=alpha),
                    jealousy=self._smooth(old_rel.jealousy, new_rel.jealousy, alpha=alpha),
                    attraction=self._smooth(old_rel.attraction, new_rel.attraction, alpha=alpha),
                )

            merged.last_updated_round = current_round
            self.long_term = merged
            return self.long_term

        except Exception:
            # 失敗しても古い long_term をそのまま返す
            return self.long_term

    # ---------------------------------------------
    # judge_mode 決定ヘルパ
    # ---------------------------------------------
    def decide_judge_mode(self, emotion: Optional[EmotionResult] = None) -> str:
        """
        短期感情（EmotionResult）＋長期感情（self.long_term）から、
        JudgeAI3 に渡すべき judge_mode を決定する。

        引数:
            emotion: 直近ターンで analyze() が返した EmotionResult。
                     None の場合は self.last_short_result を使用。

        戻り値:
            "normal" / "erotic" / "debate"
        """
        # 対象となる短期感情を決める
        if emotion is None:
            emotion = self.last_short_result

        if emotion is None:
            return "normal"

        # --- 長期側の代表値をざっくり抽出 ---
        lt = self.long_term or LongTermEmotion()
        rels = lt.relations or {}

        # 関係性の中で最も強い感情を代表値とする（かなりラフな近似）
        max_affection = 0.0
        max_attraction = 0.0
        max_anger = 0.0

        for r in rels.values():
            if r.affection > max_affection:
                max_affection = r.affection
            if r.attraction > max_attraction:
                max_attraction = r.attraction
            if r.anger > max_anger:
                max_anger = r.anger

        # --- 短期＋長期の合成（短期7 : 長期3） ---
        def mix(short_val: float, long_val: float) -> float:
            v = short_val * 0.7 + long_val * 0.3
            if v < 0.0:
                return 0.0
            if v > 1.0:
                return 1.0
            return v

        affection  = mix(emotion.affection,  max_affection)
        arousal    = mix(emotion.arousal,    max_attraction)
        tension    = mix(emotion.tension,    0.0)          # 長期緊張はまだ未定義なので 0
        anger      = mix(emotion.anger,      max_anger)
        sadness    = mix(emotion.sadness,    0.0)          # いまは mode 判定に未使用
        excitement = mix(emotion.excitement, 0.0)          # 将来 global_mood から拾ってもよい

        # --- しきい値ロジック ---

        # 1) erotic 優先
        #   - 性的昂揚が高い
        if arousal >= 0.55:
            return "erotic"

        #   - 好意＋元気が高い場合も erotic 寄り
        if affection >= 0.75 and excitement >= 0.40:
            return "erotic"

        # 2) debate 判定
        #   - 怒りが高い
        if anger >= 0.50:
            return "debate"

        #   - 緊張が高い
        if tension >= 0.65:
            return "debate"

        #   - 興奮は高いが arousal は低い → 白熱した議論モード
        if excitement >= 0.70 and arousal < 0.30:
            return "debate"

        # 3) それ以外は normal
        return "normal"
