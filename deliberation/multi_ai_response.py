# deliberation/multi_ai_response.py

from __future__ import annotations

from typing import Any, Dict, Optional

from deliberation.judge_ai import JudgeAI


class MultiAIResponse:
    """
    マルチAI関連の「ロジック中核」クラス。

    ・llm_meta を受け取り、必要なら JudgeAI を実行
    ・llm_meta に 'judge' を追加する（破壊的更新）
    ・ビュー側（DebugPanelなど）は、このクラスが整えた llm_meta を読むだけにする

    ※ここでは一切 Streamlit 等のUI処理は持たない。
    """

    def __init__(self) -> None:
        self.judge = JudgeAI()

    def process(self, llm_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        llm_meta を受け取り、JudgeAIを走らせた上で
        集約情報を返す。

        戻り値の例:
        {
            "llm_meta": <元のllm_meta（judgeが追記されている）>,
            "has_models": True/False,
            "judge": {...} or None,
        }
        """

        if not isinstance(llm_meta, dict) or not llm_meta:
            return {
                "llm_meta": None,
                "has_models": False,
                "judge": None,
            }

        models = llm_meta.get("models")
        has_models = isinstance(models, dict) and len(models) > 0

        judge_result: Optional[Dict[str, Any]] = None

        # 2モデル以上あるときだけ審判を実行
        if has_models and isinstance(models, dict) and len(models) >= 2:
            if "judge" in llm_meta and isinstance(llm_meta["judge"], dict):
                judge_result = llm_meta["judge"]
            else:
                judge_result = self.judge.run(llm_meta)
                llm_meta["judge"] = judge_result
        else:
            # そもそも比較対象が足りない
            judge_result = llm_meta.get("judge")

        return {
            "llm_meta": llm_meta,
            "has_models": has_models,
            "judge": judge_result,
        }
