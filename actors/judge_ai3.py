# actors/judge_ai3.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from actors.judge_strategy_normal import NormalJudgeStrategy
from actors.judge_strategy_erotic import EroticJudgeStrategy
from actors.judge_strategy_debate import DebateJudgeStrategy

class JudgeAI3:
    """
    複数 LLM の回答案から 1 つを選ぶ審判 AI（モード切替対応版）。

    - models_result: AnswerTalker / ModelsAI が作る llm_meta["models"] 相当
    - mode: "normal" / "erotic" / "debate" など
    """

    def __init__(self, mode: str = "normal") -> None:
        self.mode = mode
        self._strategies: Dict[str, BaseJudgeStrategy] = {
            "normal": NormalJudgeStrategy(),
            "erotic": EroticJudgeStrategy(),
            "debate": DebateJudgeStrategy(),
        }

    # ----------------------------------------
    # モード操作
    # ----------------------------------------
    def set_mode(self, mode: str) -> None:
        if mode not in self._strategies:
            raise ValueError(f"Unknown judge mode: {mode}")
        self.mode = mode

    def get_mode(self) -> str:
        return self.mode

    def _get_strategy(self) -> BaseJudgeStrategy:
        return self._strategies.get(self.mode, self._strategies["normal"])

    # ----------------------------------------
    # メイン処理
    # ----------------------------------------
    def run(
        self,
        models_result: Dict[str, Dict[str, Any]],
        context: Optional[JudgeContext] = None,
    ) -> Dict[str, Any]:
        """
        models_result: llm_meta["models"] 相当
          {
            "gpt4o": { "status": "ok", "text": "...", ... },
            "gpt51": { ... },
            ...
          }
        """
        if context is None:
            context = {}

        strategy = self._get_strategy()

        candidates: List[Dict[str, Any]] = []
        best_name: Optional[str] = None
        best_score: float = float("-inf")
        best_text: str = ""

        for name, info in models_result.items():
            if info.get("status") != "ok":
                continue

            text = (info.get("text") or "").strip()
            if not text:
                continue

            cand = JudgeCandidate(name=name, info=info, text=text)
            score, details = strategy.score_candidate(cand, context)

            entry: Dict[str, Any] = {
                "name": name,
                "status": info.get("status", "ok"),
                "score": score,
                "length": len(text),
                "text": text,
                "details": details or {},
            }
            candidates.append(entry)

            if score > best_score:
                best_score = score
                best_name = name
                best_text = text

        if best_name is None:
            return {
                "status": "no_candidate",
                "mode": self.mode,
                "chosen_model": "",
                "chosen_text": "",
                "reason": "JudgeAI3: no suitable candidate",
                "candidates": candidates,
            }

        result: Dict[str, Any] = {
            "status": "ok",
            "mode": self.mode,
            "chosen_model": best_name,
            "chosen_text": best_text,
            "reason": (
                f"JudgeAI3(mode={self.mode}) selected '{best_name}' "
                f"with score={best_score:.3f}"
            ),
            "candidates": candidates,
        }
        return result
