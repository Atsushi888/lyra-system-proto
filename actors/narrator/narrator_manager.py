# actors/narrator/narrator_manager.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from llm.llm_manager import LLMManager
# from llm.llm_manager_factory import get_llm_manager
from actors.models_ai2 import ModelsAI2
from actors.judge_ai3 import JudgeAI3


NarratorTaskType = Literal["round0", "action"]


@dataclass
class NarratorCallLog:
    """
    NarratorAI → LLM 呼び出し 1 回分のログ。
    デバッグビューで丸見えにするための構造。
    """
    task_type: NarratorTaskType
    label: str
    mode_current: str
    messages: List[Dict[str, str]]          # system / user / assistant...
    models_result: Dict[str, Any]           # ModelsAI2.collect の結果
    judge_result: Dict[str, Any]            # JudgeAI3.run の結果
    final_text: str                         # 採択されたテキスト


class NarratorManager:
    """
    ナレーション系の LLM 呼び出しを一手に引き受けるマネージャ。

    - NarratorAI は「投げたい messages」とメタ情報をここに渡すだけ。
    - ここで複数AI → JudgeAI3 の協議を行い、
      最終テキストとログを記録して返す。

    ※ emotion_override は現状 None 固定。
       将来的に『シーン感情』を直接ナレーションへ載せる場合に拡張しやすい形は維持している。
    """

    def __init__(
        self,
        persona_id: str = "narrator",
        llm_manager: Optional[LLMManager] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> None:
        # state は任意（Streamlit の session_state を渡してもよい）
        self.state: Dict[str, Any] = state if state is not None else {}

        self.llm_manager = llm_manager or LLMManager.get_or_create(person_id = persona_id)
        self.models_ai = ModelsAI2(llm_manager=self.llm_manager)
        self.judge_ai = JudgeAI3(mode="normal")

        # 呼び出し履歴
        history = self.state.get("narrator_history")
        if not isinstance(history, list):
            history = []
        self.history: List[NarratorCallLog] = history
        self.state["narrator_history"] = self.history

        # 直近の結果
        last = self.state.get("narrator_last")
        if isinstance(last, NarratorCallLog):
            self.last_result: Optional[NarratorCallLog] = last
        else:
            self.last_result = None

    # ----------------------------------------
    # メインAPI
    # ----------------------------------------
    def run_task(
        self,
        task_type: NarratorTaskType,
        label: str,
        messages: List[Dict[str, str]],
        mode_current: str = "narrator",
    ) -> str:
        """
        NarratorAI から渡された messages をもとに、
        - ModelsAI2.collect
        - JudgeAI3.run
        を実行し、最終テキストだけ返す。

        ログ（history）はデバッグビュー用に保存。
        """
        if not messages:
            return ""

        # 複数モデルから案を収集
        models_result = self.models_ai.collect(
            messages,
            mode_current=mode_current,
            emotion_override=None,
        )

        # Judge で1本選ぶ
        judge_result = self.judge_ai.run(models_result)

        final_text = (
            judge_result.get("chosen_text")
            or next(iter(models_result.values())).get("text", "")
            or ""
        ).strip()

        log_entry = NarratorCallLog(
            task_type=task_type,
            label=label,
            mode_current=mode_current,
            messages=messages,
            models_result=models_result,
            judge_result=judge_result,
            final_text=final_text,
        )

        self.history.append(log_entry)
        self.last_result = log_entry
        self.state["narrator_history"] = self.history
        self.state["narrator_last"] = log_entry

        return final_text

    # ----------------------------------------
    # ビュー用の補助メソッド
    # ----------------------------------------
    def get_history(self) -> List[NarratorCallLog]:
        return list(self.history)

    def get_last(self) -> Optional[NarratorCallLog]:
        return self.last_result
