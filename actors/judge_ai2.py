# actors/judge_ai2.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class JudgeCandidate:
    name: str
    status: str
    length: int
    priority: float
    score: float
    reason: str


class JudgeAI2:
    """
    複数モデルの回答結果 (llm_meta["models"]) から、
    どのモデルの回答を採用するかを決定するクラス。

    特徴:
      - LLM を追加で呼び出さない、ローカル評価のみ
      - priority (model_props) と text の長さを組み合わせたシンプルスコア
      - 例外は外に投げず、status="error" で返す
    """

    def __init__(self, model_props: Dict[str, Dict[str, Any]]) -> None:
        """
        Parameters
        ----------
        model_props:
            AnswerTalker から渡されるモデル定義。
            例:
            {
              "gpt4o": {"priority": 3.0, "enabled": True, ...},
              "gpt51": {"priority": 2.0, "enabled": True, ...},
              "hermes": {"priority": 1.0, "enabled": True, ...},
            }
        """
        self.model_props = model_props or {}

    # =======================
    # 公開 API
    # =======================
    def process(self, models_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        llm_meta["models"] を受け取り、採用モデルを決める。

        Parameters
        ----------
        models_result:
            ModelsAI.collect() の結果。
            形式:
            {
              "gpt4o": {"status": "ok", "text": "...", "usage": {...}, ...},
              "gpt51": {...},
              ...
            }

        Returns
        -------
        Dict[str, Any]:
            {
              "status": "ok" | "error",
              "chosen_model": str,
              "chosen_text": str,
              "reason": str,
              "candidates": [
                {
                  "name": str,
                  "status": str,
                  "length": int,
                  "priority": float,
                  "score": float,
                  "details": {...},
                },
                ...
              ],
            }
        """
        try:
            return self._safe_process(models_result)
        except Exception as e:
            # ここで例外を握りつぶし、Composer 側でフォールバックできるようにする
            return {
                "status": "error",
                "chosen_model": "",
                "chosen_text": "",
                "reason": f"JudgeAI2.process exception: {e}",
                "candidates": [],
            }

    # =======================
    # 内部実装
    # =======================
    def _safe_process(self, models_result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(models_result, dict) or not models_result:
            return {
                "status": "error",
                "chosen_model": "",
                "chosen_text": "",
                "reason": "models_result is empty or invalid",
                "candidates": [],
            }

        candidates: List[JudgeCandidate] = []

        for name, info in models_result.items():
            if not isinstance(info, dict):
                continue

            status = str(info.get("status", "unknown"))
            text = str(info.get("text") or "")
            length = len(text)

            props = self.model_props.get(name, {})
            priority_raw = props.get("priority", 1.0)
            try:
                priority = float(priority_raw)
            except Exception:
                priority = 1.0

            # status が ok で text があるものだけスコア対象
            if status == "ok" and length > 0:
                # 長すぎる / 短すぎるのバランスをとる適当な正規化
                # 例: 0〜1.5 くらいのレンジに収まるようにする
                length_norm = min(length / 500.0, 1.5)
                score = priority * length_norm
                reason = f"enabled, status=ok, len={length}, priority={priority}"
            else:
                score = 0.0
                reason = f"status={status}, len={length} => ignored for primary selection"

            cand = JudgeCandidate(
                name=name,
                status=status,
                length=length,
                priority=priority,
                score=score,
                reason=reason,
            )
            candidates.append(cand)

        # スコア順にソート（降順）
        candidates_sorted = sorted(
            candidates,
            key=lambda c: c.score,
            reverse=True,
        )

        # デバッグ用 candidates 出力用 dict に変換
        cand_dicts: List[Dict[str, Any]] = []
        for c in candidates_sorted:
            cand_dicts.append(
                {
                    "name": c.name,
                    "status": c.status,
                    "length": c.length,
                    "priority": c.priority,
                    "score": c.score,
                    "details": {
                        "reason": c.reason,
                    },
                }
            )

        # 採用候補の決定
        chosen: Optional[JudgeCandidate] = None
        for c in candidates_sorted:
            if c.status == "ok" and c.length > 0 and c.score > 0:
                chosen = c
                break

        if chosen is None:
            return {
                "status": "error",
                "chosen_model": "",
                "chosen_text": "",
                "reason": "no valid candidate (all models error/empty)",
                "candidates": cand_dicts,
            }

        chosen_info = models_result.get(chosen.name, {})
        chosen_text = str(chosen_info.get("text") or "")

        return {
            "status": "ok",
            "chosen_model": chosen.name,
            "chosen_text": chosen_text,
            "reason": f"selected by score (model={chosen.name}, score={chosen.score:.3f})",
            "candidates": cand_dicts,
        }
