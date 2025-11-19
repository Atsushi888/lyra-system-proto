# actors/judge_ai2.py

from __future__ import annotations

from typing import Any, Dict, List

from llm.llm_manager import LLMManager


class JudgeAI2:
    """
    ModelsAI が集めた各モデルの回答を評価して、
    「どのモデルのテキストを採用するか」を決めるクラス。

    - __init__ で LLMManager を受け取り、model_props 相当を参照する
    - process(models) で candidates を作り、ベストを一つ選ぶ
    """

    def __init__(self, llm_manager: LLMManager) -> None:
        self.llm_manager = llm_manager

    def _build_candidates(
        self,
        models: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []

        model_props = self.llm_manager.get_model_props()

        for name, props in model_props.items():
            info = models.get(name)

            if isinstance(props, dict):
                priority_val = props.get("priority", 1.0)
                try:
                    priority = float(priority_val)
                except Exception:
                    priority = 1.0
            else:
                priority = 1.0

            if not isinstance(info, dict):
                candidates.append(
                    {
                        "name": name,
                        "status": "status_unknown",
                        "score": 0.0,
                        "details": "status_unknown",
                        "length": 0,
                        "text": "",
                    }
                )
                continue

            status = info.get("status", "unknown")
            text = info.get("text") or ""
            length = len(text)

            if status == "ok" and text:
                length_bonus = min(length / 100.0, 10.0)
                priority_bonus = priority * 2.0
                score = length_bonus + priority_bonus
                details = (
                    f"status_ok, "
                    f"length_bonus_{length_bonus:.1f}, "
                    f"priority_bonus_{priority_bonus:.1f}"
                )
            else:
                score = 0.0
                details = f"status_ng, status_{status}"

            candidates.append(
                {
                    "name": name,
                    "status": status,
                    "score": score,
                    "details": details,
                    "length": length,
                    "text": text,
                }
            )

        return candidates

    def process(self, models: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(models, dict):
            return {
                "status": "error",
                "error": "models must be dict[str, dict]",
                "candidates": [],
            }

        candidates = self._build_candidates(models)

        if not candidates:
            return {
                "status": "error",
                "error": "no candidates",
                "candidates": [],
            }

        best = max(candidates, key=lambda c: c.get("score", 0.0))

        if not best or best.get("score", 0.0) <= 0.0:
            return {
                "status": "error",
                "error": "no valid candidates (all score <= 0)",
                "candidates": candidates,
            }

        chosen_name = best["name"]
        chosen_text = best.get("text", "")

        props_all = self.llm_manager.get_model_props()
        props = props_all.get(chosen_name, {})
        if isinstance(props, dict):
            p_raw = props.get("priority", 1.0)
            try:
                priority = float(p_raw)
            except Exception:
                priority = 1.0
        else:
            priority = 1.0

        length = best.get("length", 0)
        length_bonus = min(length / 100.0, 10.0)
        priority_bonus = priority * 2.0

        reason = (
            f"status_ok / length_bonus_{length_bonus:.1f} / "
            f"priority_bonus_{priority_bonus:.1f}"
        )

        reason_text = (
            f"{chosen_name} の出力が他候補と比べて総合スコアが最も高かったため、"
            f"このラウンドでは {chosen_name} を採用しました。"
        )

        for c in candidates:
            if "model" not in c:
                c["model"] = c.get("name", "")

        return {
            "status": "ok",
            "chosen_model": chosen_name,
            "chosen_text": chosen_text,
            "reason": reason,
            "reason_text": reason_text,
            "candidates": candidates,
        }
