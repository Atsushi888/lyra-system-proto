# actors/narrator/narrator_manager.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from llm.llm_manager import LLMManager
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
    models_result: Dict[str, Any]           # ModelsAI2.collect の結果（raw）
    judge_result: Dict[str, Any]            # JudgeAI3.run の結果
    final_text: str                         # 採択されたテキスト


class NarratorManager:
    """
    ナレーション系の LLM 呼び出しを一手に引き受けるマネージャ。

    - NarratorAI は「投げたい messages」とメタ情報をここに渡すだけ。
    - ここで複数AI → JudgeAI3 の協議を行い、最終テキストとログを記録して返す。
    - Round0 も “多AI合議制” を維持する（候補モデルは AI Manager の enabled に追従）。

    重要:
    - ModelsAI2.collect() は `_meta` / `_system` を混ぜて返すことがある。
      それをそのまま JudgeAI3 に渡すと、Judge 側の実装によっては
      「available 未定義」などの分岐を踏んで落ちることがある。
      → ここで “Judge に渡す候補” を必ず正規化してから渡す。
    """

    def __init__(
        self,
        persona_id: str = "narrator",
        llm_manager: Optional[LLMManager] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> None:
        # state は任意（Streamlit の session_state を渡してもよい）
        self.state: Dict[str, Any] = state if state is not None else {}

        self.llm_manager = llm_manager or LLMManager.get_or_create(persona_id=persona_id)
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

        # priority（将来用：NarratorAI から set_priority されるかもしれない）
        pr = self.state.get("narrator_priority")
        self._priority: List[str] = pr if isinstance(pr, list) else []

    # ----------------------------------------
    # 将来用：優先順位
    # ----------------------------------------
    def set_priority(self, priority: List[str]) -> None:
        self._priority = [str(x) for x in (priority or []) if str(x).strip()]
        self.state["narrator_priority"] = list(self._priority)

    # ----------------------------------------
    # 内部：enabled_models を解決（AI Manager 追従）
    # ----------------------------------------
    def _resolve_enabled_models(self) -> List[str]:
        """
        その時点で「呼んで良いモデル」を解決する。

        優先：
        - llm_manager.get_available_models() があれば has_key も含めて判定
        - 無ければ get_model_props() で enabled のみ判定
        - priority が設定されていれば、その順に並べ替える（未指定は末尾）
        """
        props: Dict[str, Dict[str, Any]] = {}

        if hasattr(self.llm_manager, "get_available_models"):
            try:
                props = self.llm_manager.get_available_models() or {}
            except Exception:
                props = {}
        if not props:
            try:
                # LLMManager 実装差異に備えて両方試す
                if hasattr(self.llm_manager, "get_model_props"):
                    props = self.llm_manager.get_model_props() or {}
                elif hasattr(self.llm_manager, "get_model_props"):
                    props = self.llm_manager.get_model_props() or {}
            except Exception:
                props = {}

        enabled: List[str] = []
        for name, p in (props or {}).items():
            try:
                if not isinstance(p, dict):
                    continue
                if not p.get("enabled", True):
                    continue
                # get_available_models() 系は has_key が入る
                if "has_key" in p and not p.get("has_key", True):
                    continue
                enabled.append(str(name))
            except Exception:
                enabled.append(str(name))

        # priority を反映（指定されたものを先頭に）
        if self._priority:
            pri = [m for m in self._priority if m in enabled]
            rest = [m for m in enabled if m not in pri]
            enabled = pri + rest

        return enabled

    # ----------------------------------------
    # 内部：ModelsAI2 の戻りを Judge 用に正規化
    # ----------------------------------------
    @staticmethod
    def _extract_judge_candidates(models_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        ModelsAI2.collect() の戻りから、JudgeAI3 に渡して安全な形だけ抜く。

        - "_meta" / "_system" など "_" で始まるキーは除外
        - 値が dict で、最低限 "status"/"text" を持つものを優先
        """
        out: Dict[str, Dict[str, Any]] = {}

        for name, info in (models_result or {}).items():
            if not isinstance(name, str):
                continue
            if name.startswith("_"):
                continue
            if not isinstance(info, dict):
                continue

            # できるだけ揃える（Judge 側の想定差異を吸収）
            status = info.get("status", "unknown")
            text = info.get("text", "") or ""

            out[name] = {
                **info,
                "status": status,
                "text": text,
            }

        return out

    @staticmethod
    def _pick_first_text(cands: Dict[str, Dict[str, Any]]) -> str:
        """
        Judge が失敗した場合のフォールバック。
        """
        for _, info in (cands or {}).items():
            if not isinstance(info, dict):
                continue
            t = (info.get("text") or "").strip()
            if t:
                return t
        return ""

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

        # ✅ その時点の enabled_models を確定（AI Manager のON/OFFに追従）
        enabled_models = self._resolve_enabled_models()

        # ✅ 多AI合議制は維持：enabled_models が複数なら複数、1個なら「1個で合議」
        #    （構造は維持、候補集合だけが変わる）
        models_ai = ModelsAI2(
            llm_manager=self.llm_manager,
            enabled_models=enabled_models if enabled_models else None,
        )

        # 複数モデルから案を収集（raw）
        models_result = models_ai.collect(
            messages,
            mode_current=mode_current,
            emotion_override=None,
        )

        # ✅ Judge に渡す候補を正規化（"_meta" 等を混ぜない）
        judge_candidates = self._extract_judge_candidates(models_result)

        # Judge で1本選ぶ（Judge 側が候補ゼロで落ちる実装でも、ここで守る）
        judge_result: Dict[str, Any] = {}
        chosen_text = ""

        if judge_candidates:
            try:
                judge_result = self.judge_ai.run(judge_candidates)
                chosen_text = (judge_result.get("chosen_text") or "").strip()
            except Exception as e:
                judge_result = {
                    "status": "error",
                    "error": str(e),
                    "chosen_text": "",
                    "chosen_model": "",
                    "reason": "JudgeAI3 failed in NarratorManager",
                }
        else:
            judge_result = {
                "status": "error",
                "error": "No judge candidates (all filtered or empty)",
                "chosen_text": "",
                "chosen_model": "",
                "reason": "ModelsAI2 returned no usable candidates",
            }

        # 最終テキスト（Judge優先、ダメなら候補の先頭から）
        final_text = chosen_text or self._pick_first_text(judge_candidates)
        final_text = (final_text or "").strip()

        log_entry = NarratorCallLog(
            task_type=task_type,
            label=label,
            mode_current=mode_current,
            messages=messages,
            models_result=models_result,      # raw を保存（デバッグ重要）
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
