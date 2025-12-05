# actors/emotion_ai.py
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
import json

from llm.llm_manager import LLMManager
from actors.emotion_modes.context import JudgeSignal, get_default_selectors


# ==============================
# データ構造（短期・長期）
# ==============================

@dataclass
class EmotionResult:
    """
    フローリア（および将来のキャラ）自身の
    「短期的な感情状態」を表すスナップショット。
    （1ターンごと / ComposerAI の最終返答ベース）
    """
    mode: str = "normal"        # "normal" / "erotic" / "debate" など
    affection: float = 0.0      # 好意・親しみ
    arousal: float = 0.0        # 性的な高ぶり
    tension: float = 0.0        # 緊張・不安
    anger: float = 0.0          # 怒り
    sadness: float = 0.0        # 悲しみ
    excitement: float = 0.0     # 期待・ワクワク
    raw_text: str = ""          # LLM の生返答（JSONそのもの or エラー）

    # ---- SuperPack 試験運用フィールド（ドキドキ💓パワー） ----
    doki_power: float = 0.0     # ドキドキ💓パワー 0.0〜100.0 想定
    doki_level: int = 0         # ドキドキ段階 0〜3 くらい
    meta: Dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------
    # 辞書⇄オブジェクト 変換まわり
    # ----------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        デバッグ／保存用の辞書化。
        affection_with_doki などの派生値も含める。
        """
        data = asdict(self)
        data["affection_with_doki"] = self.affection_with_doki
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmotionResult":
        """
        将来のセーブデータ復元用。
        足りない項目はデフォルト値で補完する。
        """
        if not isinstance(data, dict):
            return cls()

        return cls(
            mode=str(data.get("mode", "normal")),
            affection=float(data.get("affection", 0.0)),
            arousal=float(data.get("arousal", 0.0)),
            tension=float(data.get("tension", 0.0)),
            anger=float(data.get("anger", 0.0)),
            sadness=float(data.get("sadness", 0.0)),
            excitement=float(data.get("excitement", 0.0)),
            raw_text=str(data.get("raw_text", "")),

            doki_power=float(data.get("doki_power", 0.0)),
            doki_level=int(data.get("doki_level", 0)),
            meta=dict(data.get("meta") or {}),
        )

    # ----------------------------------------
    # ドキドキ💓補正ロジック
    # ----------------------------------------
    @property
    def affection_with_doki(self) -> float:
        """
        ドキドキ💓パワーによる「ゲタ」を履かせた実効好感度。
        いまは暫定的に doki_level ごとの固定ボーナスでシンプルに定義。

        - level 0: +0.0
        - level 1: +0.15
        - level 2: +0.35
        - level 3: +0.60

        ※将来的に SuperPack 側でチューニングしてよい前提。
        """
        bonus_table = [0.0, 0.15, 0.35, 0.60]
        idx = max(0, min(self.doki_level, len(bonus_table) - 1))
        bonus = bonus_table[idx]

        base = max(0.0, min(1.0, self.affection))
        return min(1.0, base + bonus)


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
    global_mood: Dict[str, float] = field(default_factory=dict)
    relations: Dict[str, RelationEmotion] = field(default_factory=dict)
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

    - analyze():          短期的な感情（EmotionResult）を推定
    - update_long_term(): MemoryRecord 群から長期感情 LongTermEmotion を更新
    - decide_judge_mode():短期＋長期の感情から judge_mode を決定
                          （内部で Strategy クラス群に委譲）
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

        # judge_mode 決定用 Strategy 群
        self._selectors = get_default_selectors()

    # ---------------------------------------------
    # 短期感情：Composer + memory_context ベース
    # ---------------------------------------------
    def _build_messages(
        self,
        composer: Dict[str, Any],
        memory_context: str = "",
        user_text: str = "",
    ) -> List[Dict[str, str]]:
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
                max_completion_tokens=320,  # gpt-5.1 用
            )

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
  "global_mood": { ... },
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
    ...
  }
}
"""
        lines: List[str] = []
        lines.append("=== Important memories of Floria ===")

        for idx, rec in enumerate(memory_records, start=1):
            summary = getattr(rec, "summary", None)
            importance = getattr(rec, "importance", None)
            tags = getattr(rec, "tags", None)
            round_id = getattr(rec, "round_id", None)

            if summary is None and isinstance(rec, dict):
                summary = rec.get("summary")
                importance = rec.get("importance")
                tags = rec.get("tags")
                round_id = rec.get("round_id")

            summary = summary or ""
            imp = importance if importance is not None else "?"
            tag_str = ", ".join(tags) if tags else ""
            rid = round_id if round_id is not None else "?"

            lines.append(
                f"- #{idx} (round={rid}, importance={imp}, tags=[{tag_str}]): {summary}"
            )

        user_prompt = "\n".join(lines)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _smooth(self, old: float, new: float, alpha: float = 0.3) -> float:
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
        """
        if not memory_records:
            return self.long_term

        messages = self._build_long_term_messages(memory_records)

        try:
            raw = self.llm_manager.call_model(
                model_name=self.model_name,
                messages=messages,
                temperature=0.1,
                max_completion_tokens=512,
            )

            if isinstance(raw, tuple):
                text = raw[0]
            else:
                text = raw

            if not isinstance(text, str):
                text = "" if text is None else str(text)

            data = json.loads(text)
            parsed = LongTermEmotion.from_dict(data)

            merged = LongTermEmotion()
            merged.last_updated_round = (
                current_round or self.long_term.last_updated_round
            )

            # global_mood
            all_mood_keys = set(self.long_term.global_mood.keys()) | set(
                parsed.global_mood.keys()
            )
            for key in all_mood_keys:
                old_val = self.long_term.global_mood.get(key, 0.0)
                new_val = parsed.global_mood.get(key, 0.0)
                merged.global_mood[key] = self._smooth(old_val, new_val, alpha=alpha)

            # relations
            all_rel_keys = set(self.long_term.relations.keys()) | set(
                parsed.relations.keys()
            )
            for rel_name in all_rel_keys:
                old_rel = self.long_term.relations.get(rel_name, RelationEmotion())
                new_rel = parsed.relations.get(rel_name, RelationEmotion())

                merged.relations[rel_name] = RelationEmotion(
                    affection=self._smooth(
                        old_rel.affection, new_rel.affection, alpha=alpha
                    ),
                    trust=self._smooth(old_rel.trust, new_rel.trust, alpha=alpha),
                    anger=self._smooth(old_rel.anger, new_rel.anger, alpha=alpha),
                    fear=self._smooth(old_rel.fear, new_rel.fear, alpha=alpha),
                    sadness=self._smooth(old_rel.sadness, new_rel.sadness, alpha=alpha),
                    jealousy=self._smooth(
                        old_rel.jealousy, new_rel.jealousy, alpha=alpha
                    ),
                    attraction=self._smooth(
                        old_rel.attraction, new_rel.attraction, alpha=alpha
                    ),
                )

            merged.last_updated_round = current_round
            self.long_term = merged
            return self.long_term

        except Exception:
            return self.long_term

    # ---------------------------------------------
    # judge_mode 決定ヘルパ（Strategy に委譲）
    # ---------------------------------------------
    def decide_judge_mode(self, emotion: Optional[EmotionResult] = None) -> str:
        """
        短期感情（EmotionResult）＋長期感情（self.long_term）から、
        JudgeAI3 に渡すべき judge_mode を決定する。

        - 長期感情がまだ空の段階では、短期感情の mode をそのまま採用
        - それ以降は JudgeSignal を作り、Strategy 群に判定を委譲
        """
        # 対象となる短期感情を決める
        if emotion is None:
            emotion = self.last_short_result

        if emotion is None:
            return "normal"

        # 1) long_term がまだ一度も更新されていない → raw の mode を信用
        if (
            (not self.long_term.global_mood)
            and (not self.long_term.relations)
            and self.long_term.last_updated_round == 0
        ):
            # erotic 判定が出ているのに normal に潰される問題はここで防ぐ
            return emotion.mode or "normal"

        # 2) 長期側の代表値をざっくり抽出
        lt = self.long_term or LongTermEmotion()
        rels = lt.relations or {}

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

        # 3) 短期 × 1.0 ＋ 長期 × 0.2
        def mix(short_val: float, long_val: float) -> float:
            v = short_val * 1.0 + long_val * 0.2
            if v < 0.0:
                return 0.0
            if v > 1.0:
                return 1.0
            return v

        affection  = mix(emotion.affection,  max_affection)
        arousal    = mix(emotion.arousal,    max_attraction)
        tension    = mix(emotion.tension,    0.0)
        anger      = mix(emotion.anger,      max_anger)
        sadness    = mix(emotion.sadness,    0.0)
        excitement = mix(emotion.excitement, 0.0)

        # 4) JudgeSignal を構築して Strategy 群に渡す
        signal = JudgeSignal(
            short_mode=emotion.mode or "normal",
            affection=affection,
            arousal=arousal,
            tension=tension,
            anger=anger,
            sadness=sadness,
            excitement=excitement,
        )

        # 優先度順に Selector を適用
        for selector in self._selectors:
            mode = selector.select(signal)
            if mode:
                return mode

        # どれも選ばなければ normal
        return "normal"
