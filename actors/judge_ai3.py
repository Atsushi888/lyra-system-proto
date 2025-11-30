# actors/judge_ai3.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class JudgeCandidate:
    name: str
    text: str
    status: str
    length: int
    base_score: float
    length_score: float
    total_score: float


class JudgeAI3:
    """
    複数モデルの回答候補から「どれを採用するか」を決める審査クラス（第3世代）。

    方針:
      - ModelsAI2 が集めた llm_meta["models"] を入力として評価する
      - プレイヤーの発話長 (user_text の文字数) を参考に
          - ユーザー文が短い → やや長めの回答を好む
          - ユーザー文が長い → やや短めの回答を好む
        という「長さの嗜好」を導入する
      - 将来的に emotion_mode などもスコア式に足せるよう、内部は加点方式で設計

    期待される models 構造:
      models = {
          "gpt51": {"status": "ok", "text": "...", ...},
          "grok":  {"status": "ok", "text": "...", ...},
          ...
      }

    run() 戻り値の例:
      {
          "status": "ok",
          "mode": "normal",
          "chosen_model": "gpt51",
          "chosen_text": "・・・",
          "reason": "...",
          "candidates": [
              {
                  "name": "gpt51",
                  "score": 0.83,
                  "length": 240,
                  "length_score": 0.9,
                  "base_score": 0.8,
                  "details": { ... },
              },
              ...
          ],
      }
    """

    def __init__(self, mode: str = "normal") -> None:
        self.mode = mode  # "normal" / "erotic" / "debate" など

    # --------------------------------------------------
    # 外部 API
    # --------------------------------------------------
    def set_mode(self, mode: str) -> None:
        """外部から Judge モードを差し替えるための簡易 setter。"""
        self.mode = mode or "normal"

    def run(
        self,
        models: Dict[str, Any],
        *,
        user_text: str = "",
    ) -> Dict[str, Any]:
        """
        models: llm_meta["models"]
        user_text: このターンのプレイヤー発話。長さスコア算出の材料に使う。
        """
        # 候補生成
        candidates = self._collect_candidates(models, user_text=user_text)

        if not candidates:
            return {
                "status": "error",
                "mode": self.mode,
                "chosen_model": "",
                "chosen_text": "",
                "reason": "[JudgeAI3] no valid candidates",
                "candidates": [],
            }

        # スコア最大の候補を選択
        best = max(
            candidates,
            key=lambda c: (c.total_score, c.base_score, c.length),
        )

        # デバッグ用に候補一覧を dict 化
        cand_list: List[Dict[str, Any]] = []
        for c in candidates:
            cand_list.append(
                {
                    "name": c.name,
                    "score": round(c.total_score, 4),
                    "length": c.length,
                    "length_score": round(c.length_score, 4),
                    "base_score": round(c.base_score, 4),
                    "details": {
                        "status": c.status,
                        "length": c.length,
                        "length_score": c.length_score,
                        "base_score": c.base_score,
                        "total_score": c.total_score,
                    },
                }
            )

        reason = self._build_reason(best, user_text=user_text)

        return {
            "status": "ok",
            "mode": self.mode,
            "chosen_model": best.name,
            "chosen_text": best.text,
            "reason": reason,
            "candidates": cand_list,
        }

    # --------------------------------------------------
    # 内部実装
    # --------------------------------------------------
    def _collect_candidates(
        self,
        models: Dict[str, Any],
        *,
        user_text: str,
    ) -> List[JudgeCandidate]:
        """
        models dict から評価対象となる候補を集め、スコアリングして返す。
        """
        if not isinstance(models, dict):
            return []

        user_len = len(user_text or "")
        target_len = self._calc_preferred_length(user_len=user_len)

        # モデルごとの「基本優先度」（tie-breaker 用）
        # 必要に応じて調整可
        base_priority_order = ["gpt51", "gpt4o", "grok", "gemini", "hermes"]
        base_priority_map = {
            name: (len(base_priority_order) - idx) / float(len(base_priority_order))
            for idx, name in enumerate(base_priority_order)
        }

        candidates: List[JudgeCandidate] = []

        for name, info in models.items():
            if not isinstance(info, dict):
                continue

            status = str(info.get("status") or "unknown")
            if status != "ok":
                continue

            text = str(info.get("text") or "").strip()
            if not text:
                continue

            length = len(text)

            # ---- 1) ベーススコア（今はほぼモデル優先度のみ）----
            base_score = base_priority_map.get(name, 0.5)

            # ---- 2) 長さの嗜好スコア ----
            length_score = self._length_preference_score(
                answer_len=length,
                target_len=target_len,
            )

            # ---- 3) 総合スコア ----
            #   - length_score をやや重めに扱う
            total_score = (base_score * 0.4) + (length_score * 0.6)

            candidates.append(
                JudgeCandidate(
                    name=name,
                    text=text,
                    status=status,
                    length=length,
                    base_score=base_score,
                    length_score=length_score,
                    total_score=total_score,
                )
            )

        return candidates

    # --------------------------------------------------
    # 長さまわりのロジック
    # --------------------------------------------------
    def _calc_preferred_length(self, *, user_len: int) -> int:
        """
        プレイヤー発話長から「好ましい回答長の目安」を計算する。

        イメージ:
          - user_len が短い (例: 0〜50)  →  だいたい 260 前後のやや長めを好む
          - user_len が長い (例: 300 以上) → だいたい 120 前後のやや短めを好む

        線形補間でシンプルに決めているだけなので、必要に応じて調整可。
        """
        # ユーザー文の長さを 0〜1 に正規化（300 文字以上は 1 扱い）
        u = max(0.0, min(1.0, user_len / 300.0))

        # u=0 のときのターゲット（ユーザーが短文）→ 長め
        target_long = 260  # 好みで調整可

        # u=1 のときのターゲット（ユーザーが長文）→ 短め
        target_short = 120

        target = int(round(target_long * (1.0 - u) + target_short * u))
        return max(60, target)  # あまりに短くならないように下限を設定

    def _length_preference_score(self, *, answer_len: int, target_len: int) -> float:
        """
        「回答長がターゲット長にどれくらい近いか」を 0.0〜1.0 でスコアリングする。

        diff_ratio = |answer_len - target_len| / target_len
        をもとに、diff_ratio が 0 のとき 1.0、1.0 のとき 0.0 になるような
        緩やかなスコアにしている。
        """
        if target_len <= 0:
            return 0.5

        diff = abs(answer_len - target_len)
        diff_ratio = diff / float(target_len)

        # 差が 0 → 1.0, 差が target_len → 0.0
        raw = 1.0 - diff_ratio
        # 多少ゆるめにしておく（必要なら係数を調整）
        score = max(0.0, min(1.0, raw))
        return score

    # --------------------------------------------------
    # 説明テキスト生成
    # --------------------------------------------------
    def _build_reason(self, best: JudgeCandidate, *, user_text: str) -> str:
        """
        選択理由の要約を作る（デバッグ＆説明用）。
        """
        user_len = len(user_text or "")
        target_len = self._calc_preferred_length(user_len=user_len)

        return (
            f"[JudgeAI3] mode={self.mode}, "
            f"user_len={user_len}, "
            f"target_len≈{target_len}, "
            f"chosen={best.name} "
            f"(len={best.length}, base_score={best.base_score:.3f}, "
            f"length_score={best.length_score:.3f}, "
            f"total={best.total_score:.3f})"
        )
