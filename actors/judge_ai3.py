# actors/judge_ai3.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class JudgeCandidate:
    """
    JudgeAI3 が内部で使うスコア付き候補。
    """
    name: str
    text: str
    length: int
    score: float
    details: Dict[str, Any]


class JudgeAI3:
    """
    複数 LLM の回答候補（llm_meta["models"]）から
    「どのモデルのテキストを採用するか」を決める審判クラス。

    v0.3 方針:
      - 文章の長さだけで決めない
      - 長さ・フォーマット・NG表現など複数指標でスコアリング
      - 将来的な拡張（Scene / Persona 連携）を見据えて構造はシンプルに
    """

    def __init__(self, mode: str = "normal") -> None:
        self._mode: str = (mode or "normal").lower()

    # ----------------------------------------------------
    # 公開 API
    # ----------------------------------------------------
    def set_mode(self, mode: str) -> None:
        self._mode = (mode or "normal").lower()

    @property
    def mode(self) -> str:
        return self._mode

    # models: AnswerTalker.llm_meta["models"]
    def run(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """
        models から最適な 1 本を選ぶ。

        Parameters
        ----------
        models: Dict[str, Any]
            ModelsAI2.collect() が返した dict を想定:
            {
              "gpt51": {"status": "ok", "text": "...", "usage": {...}, "meta": {...}},
              "grok":  {...},
              ...
            }

        Returns
        -------
        Dict[str, Any] 例:
            {
              "status": "ok",
              "chosen_model": "gpt51",
              "chosen_text": "...",
              "reason": "スコアに基づき gpt51 を選択しました ...",
              "candidates": [
                 {"name": "gpt51", "score": 87.5, "length": 320, "details": {...}},
                 {"name": "grok",  "score": 75.2, ...},
                 ...
              ]
            }
        """
        try:
            return self._safe_run(models)
        except Exception as e:
            return {
                "status": "error",
                "error": f"[JudgeAI3] exception: {e}",
                "chosen_model": "",
                "chosen_text": "",
                "candidates": [],
            }

    # ----------------------------------------------------
    # 内部実装
    # ----------------------------------------------------
    def _safe_run(self, models: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(models, dict) or not models:
            return {
                "status": "error",
                "error": "[JudgeAI3] models が空です。",
                "chosen_model": "",
                "chosen_text": "",
                "candidates": [],
            }

        candidates: List[JudgeCandidate] = []

        for name, info in models.items():
            if not isinstance(info, dict):
                continue

            status = str(info.get("status", ""))
            if status != "ok":
                continue

            text = str(info.get("text") or "").strip()
            if not text:
                continue

            length = len(text)

            # 各種スコア算出
            length_score, len_detail = self._score_length(length)
            fmt_score, fmt_detail = self._score_format(text)
            ng_penalty, ng_detail = self._score_ng_phrases(text)

            total_score = length_score + fmt_score + ng_penalty

            cand = JudgeCandidate(
                name=name,
                text=text,
                length=length,
                score=total_score,
                details={
                    "length_score": length_score,
                    "format_score": fmt_score,
                    "ng_penalty": ng_penalty,
                    "length_detail": len_detail,
                    "format_detail": fmt_detail,
                    "ng_detail": ng_detail,
                },
            )
            candidates.append(cand)

        if not candidates:
            return {
                "status": "error",
                "error": "[JudgeAI3] 有効な候補テキストがありません。",
                "chosen_model": "",
                "chosen_text": "",
                "candidates": [],
            }

        # スコア順にソート（降順）
        candidates.sort(key=lambda c: c.score, reverse=True)
        best = candidates[0]

        # 外部に渡すためにシリアライズ
        cand_list = [
            {
                "name": c.name,
                "score": round(c.score, 2),
                "length": c.length,
                "details": c.details,
            }
            for c in candidates
        ]

        reason = self._build_reason(best, cand_list)

        return {
            "status": "ok",
            "chosen_model": best.name,
            "chosen_text": best.text,
            "reason": reason,
            "candidates": cand_list,
        }

    # ----------------------------------------------------
    # スコアリング要素
    # ----------------------------------------------------
    def _score_length(self, length: int) -> Tuple[float, Dict[str, Any]]:
        """
        文章の長さに基づいてスコアリング。
        - 短すぎても長すぎても減点。
        """
        # 目安レンジ
        ideal_min = 80      # この辺から「ちゃんとした返答」
        ideal_max = 600     # これを超えるとダラダラし始める
        hard_min = 40       # これ未満はほぼ一言
        hard_max = 1200     # ここを超えるのはさすがに長すぎ

        if length <= 0:
            return -100.0, {"msg": "empty"}

        # 基本点
        base = 50.0

        # 短すぎペナルティ
        if length < hard_min:
            base -= 40.0
        elif length < ideal_min:
            # 軽めの減点（ちょっと短い）
            base -= (ideal_min - length) * 0.2

        # 長すぎペナルティ
        if length > hard_max:
            base -= 40.0
        elif length > ideal_max:
            base -= (length - ideal_max) * 0.05

        # 最小値・最大値を軽くクリップ
        base = max(-100.0, min(base, 60.0))

        detail = {
            "length": length,
            "ideal_min": ideal_min,
            "ideal_max": ideal_max,
            "hard_min": hard_min,
            "hard_max": hard_max,
        }
        return base, detail

    def _score_format(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """
        箇条書き・Markdown・装飾記号などを検出して減点。
        Lyra-System の「素の日本語の文章」方針に近いほど高評価。
        """
        lines = text.splitlines()
        penalty = 0.0
        bullet_starts = ("*", "・", "-", "#", "★")

        bullet_lines = 0
        markdown_hits = 0

        for line in lines:
            l = line.strip()
            if not l:
                continue
            if l.startswith(bullet_starts):
                bullet_lines += 1
            if "```" in l or "###" in l:
                markdown_hits += 1

        if bullet_lines > 0:
            penalty -= min(20.0, bullet_lines * 3.0)
        if markdown_hits > 0:
            penalty -= min(20.0, markdown_hits * 5.0)

        # ベースは +20 くらいから開始して、ペナルティを引く
        score = 20.0 + penalty
        score = max(-40.0, min(score, 25.0))

        detail = {
            "lines": len(lines),
            "bullet_lines": bullet_lines,
            "markdown_hits": markdown_hits,
        }
        return score, detail

    def _score_ng_phrases(self, text: str) -> Tuple[float, Dict[str, Any]]:
        """
        「AI 言語モデルです」「申し訳ありません」など
        物語世界から浮く説明・謝罪・拒否表現を検出して減点。
        """
        lower = text.lower()
        ng_patterns = [
            "ai 言語モデル",
            "aiモデル",
            "ai のモデル",
            "申し訳ありません",
            "申し訳ございません",
            "すみませんが",
            "対応できません",
            "お答えできません",
            "制限されています",
            "ガイドライン",
        ]

        hits: List[str] = []
        penalty = 0.0

        for pat in ng_patterns:
            if pat in text or pat in lower:
                hits.append(pat)
                penalty -= 8.0

        # ペナルティは最大 -40 に抑える
        penalty = max(-40.0, penalty)

        detail = {
            "ng_hits": hits,
        }
        return penalty, detail

    # ----------------------------------------------------
    # 理由文生成
    # ----------------------------------------------------
    def _build_reason(
        self,
        best: JudgeCandidate,
        all_cands: List[Dict[str, Any]],
    ) -> str:
        """
        開発者向けに簡単なサマリを返す。
        """
        msg = (
            f"[JudgeAI3] mode={self._mode} で評価を実施。\n"
            f"選択モデル: {best.name} / score={best.score:.2f} / length={best.length}\n"
            f"- length_score: {best.details.get('length_score'):.2f}\n"
            f"- format_score: {best.details.get('format_score'):.2f}\n"
            f"- ng_penalty:   {best.details.get('ng_penalty'):.2f}\n"
            "\n"
            "他候補のスコアも llm_meta['judge']['candidates'] に格納されています。"
        )
        return msg
