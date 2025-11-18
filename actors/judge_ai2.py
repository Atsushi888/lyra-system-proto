# actors/judge_ai2.py

from __future__ import annotations
from typing import Any, Dict, List


class JudgeAI2:
    def __init__(self, model_props: Dict[str, Dict[str, Any]]) -> None:
        self.model_props = model_props or {}

        # priority の高い順に並べたモデル名リストを作る
        # priority 未指定は 0 扱い
        items: List[tuple[str, int]] = []
        for name, props in self.model_props.items():
            prio = int(props.get("priority", 0))
            items.append((name, prio))

        # prio 降順でソート
        items.sort(key=lambda x: x[1], reverse=True)
        self.model_priority: List[str] = [name for name, _ in items]

    def process(self, llm_meta: Dict[str, Any]) -> Dict[str, Any]:
        models = llm_meta.get("models") or {}
        if not isinstance(models, dict) or not models:
            return {
                "status": "no_candidate",
                "chosen_model": "",
                "chosen_text": "",
                "reason": "models が空です",
                "candidates": [],
            }

        candidates: List[Dict[str, Any]] = []

        for name, info in models.items():
            if not isinstance(info, dict):
                continue

            status = info.get("status", "unknown")
            text = (info.get("text") or "").strip()
            error = info.get("error")

            # disabled はスキップ
            if status == "disabled":
                continue

            score = 0.0
            details: List[str] = []

            if status == "ok":
                score += 10.0
                details.append("status_ok")
            else:
                details.append(f"status_{status}")
                candidates.append({
                    "model": name,
                    "status": status,
                    "text": text,
                    "score": -9999.0,
                    "length": len(text),
                    "error": error,
                    "details": details + ["excluded_from_choice"],
                })
                continue

            length = len(text)
            if not text:
                score -= 5.0
                details.append("empty_text")
            else:
                length_bonus = min(length / 100.0, 10.0)
                score += length_bonus
                details.append(f"length_bonus_{length_bonus:.1f}")

            if name in self.model_priority:
                idx = self.model_priority.index(name)
                prio_bonus = (len(self.model_priority) - idx) * 2.0
                score += prio_bonus
                details.append(f"priority_bonus_{prio_bonus:.1f}")
            else:
                details.append("priority_unknown")

            candidates.append({
                "model": name,
                "status": status,
                "text": text,
                "score": score,
                "length": length,
                "error": error,
                "details": details,
            })

        effective = [c for c in candidates if "excluded_from_choice" not in c["details"]]
        if not effective:
            return {
                "status": "no_candidate",
                "chosen_model": "",
                "chosen_text": "",
                "reason": "有効な候補モデルがありません",
                "candidates": candidates,
            }

        effective.sort(key=lambda c: c["score"], reverse=True)
        chosen = effective[0]

        return {
            "status": "ok",
            "chosen_model": chosen["model"],
            "chosen_text": chosen["text"],
            "reason": " / ".join(chosen["details"]),
            "candidates": candidates,
        }
