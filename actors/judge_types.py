# actors/judge_types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


JudgeContext = Dict[str, Any]


@dataclass
class JudgeCandidate:
    """
    1つの候補回答を表すデータコンテナ。
    """
    name: str               # モデル名（"gpt4o" など）
    info: Dict[str, Any]    # llm_meta["models"][name] 相当
    text: str               # 実際の回答テキスト


class BaseJudgeStrategy:
    """
    1つの「審査モード」を表す抽象クラス。

    - mode_name: "normal" / "erotic" / "debate" などの識別子
    - score_candidate: 候補にスコアを付ける本体メソッド
      戻り値は (score, details) のタプルで、
      details は AnswerTalkerView から参照できる任意情報。
    """

    mode_name: str = "base"

    def score_candidate(
        self,
        cand: JudgeCandidate,
        context: JudgeContext,
    ) -> Tuple[float, Dict[str, Any]]:
        raise NotImplementedError
