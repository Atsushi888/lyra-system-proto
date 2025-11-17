# actors/judge_ai2.py

from __future__ import annotations
from typing import Any, Dict

# ★ AnswerTalker を import しない！！
# from actors.answer_talker import AnswerTalker  ← これがあるとアウト

class JudgeAI2:
    def __init__(self) -> None:
        pass

    def process(self, llm_meta: Dict[str, Any]) -> Dict[str, Any]:
        # ひとまずダミー実装でOK
        models = llm_meta.get("models") or {}
        return {
            "chosen_model": "gpt4o" if "gpt4o" in models else "",
            "reason": "dummy",
        }
