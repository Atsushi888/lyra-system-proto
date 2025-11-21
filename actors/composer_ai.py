# actors/composer_ai.py
from __future__ import annotations

from typing import Any, Dict, Optional

from llm.llm_manager import LLMManager


class ComposerAI:
    """
    llm_meta["models"] と llm_meta["judge"] をもとに、
    「最終返答テキスト」を組み立てるクラス。

    v0.2.x の方針:
      - 原則として追加で LLM を呼ばない（安定性優先）
      - JudgeAI2 の chosen_text を優先採用
      - Judge が error の場合は models からフォールバック
      - 将来的に llm_manager を渡して refinement を有効化できる設計
    """

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        refine_model: str = "gpt4o",
    ) -> None:
        """
        Parameters
        ----------
        llm_manager:
            追加の refinement に使う LLMManager。
            いまは None を指定して、LLM呼び出しなし運用でも OK。

        refine_model:
            refinement に使うモデル名（llm_manager 管理下の名前）。
        """
        self.llm_manager = llm_manager
        self.refine_model = refine_model

    # =======================
    # 公開 API
    # =======================
    def compose(self, llm_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        llm_meta 全体を受け取り、最終返答テキストとメタ情報を返す。

        Returns
        -------
        Dict[str, Any]:
            {
              "status": "ok" | "error",
              "text": str,
              "source_model": str,
              "mode": "judge_choice" | "fallback_from_models" | "refined",
              "summary": str,
            }
        """
        try:
            return self._safe_compose(llm_meta)
        except Exception as e:
            return {
                "status": "error",
                "text": "",
                "source_model": "",
                "mode": "exception",
                "summary": f"[ComposerAI] exception: {e}",
            }

    # =======================
    # 内部実装
    # =======================
    def _safe_compose(self, llm_meta: Dict[str, Any]) -> Dict[str, Any]:
        models = llm_meta.get("models") or {}
        judge = llm_meta.get("judge") or {}

        # 1) JudgeAI2 の結果を優先
        judge_status = str(judge.get("status", ""))
        chosen_model = str(judge.get("chosen_model") or "")
        chosen_text = str(judge.get("chosen_text") or "")

        if judge_status == "ok" and chosen_model and chosen_text.strip():
            text = chosen_text
            mode = "judge_choice"
            source_model = chosen_model
            summary = (
                "[ComposerAI 0.2.x] mode=judge_choice, "
                f"source_model={source_model}, judge_status=ok"
            )
            # (必要であればここで refinement を噛ませる)
            return self._maybe_refine(
                base=dict(
                    status="ok",
                    text=text,
                    source_model=source_model,
                    mode=mode,
                    summary=summary,
                ),
                llm_meta=llm_meta,
            )

        # 2) Judge がエラー or 空のときは models からフォールバック
        fallback_model, fallback_text = self._fallback_from_models(models)

        if fallback_model and fallback_text.strip():
            text = fallback_text
            mode = "fallback_from_models"
            source_model = fallback_model
            summary = (
                "[ComposerAI 0.2.x] mode=fallback_from_models, "
                f"source_model={source_model}, judge_status={judge_status}"
            )
            return self._maybe_refine(
                base=dict(
                    status="ok",
                    text=text,
                    source_model=source_model,
                    mode=mode,
                    summary=summary,
                ),
                llm_meta=llm_meta,
            )

        # 3) それでも何もない場合
        return {
            "status": "error",
            "text": "",
            "source_model": "",
            "mode": "no_text",
            "summary": "[ComposerAI 0.2.x] no usable text from judge or models",
        }

    # -----------------------
    # モデルからのフォールバック
    # -----------------------
    @staticmethod
    def _fallback_from_models(models: Dict[str, Any]) -> tuple[str, str]:
        """
        models から、もっとも無難なテキストを 1 つ選ぶ。

        いまは「status=ok かつ text が空でない最初のモデル」を返す。
        必要であれば優先度順に見るよう拡張可能。
        """
        if not isinstance(models, dict):
            return "", ""

        # まず gpt4o / gpt51 / hermes の順に見る（好みの順で OK）
        preferred_order = ["gpt4o", "gpt51", "hermes"]

        # 優先モデルからチェック
        for name in preferred_order:
            info = models.get(name)
            if not isinstance(info, dict):
                continue
            if info.get("status") != "ok":
                continue
            text = str(info.get("text") or "").strip()
            if text:
                return name, text

        # どれでも良いので最初の ok を返す
        for name, info in models.items():
            if not isinstance(info, dict):
                continue
            if info.get("status") != "ok":
                continue
            text = str(info.get("text") or "").strip()
            if text:
                return name, text

        return "", ""

    # -----------------------
    # （任意）refinement
    # -----------------------
    def _maybe_refine(
        self,
        base: Dict[str, Any],
        llm_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        将来的に gpt-5.1 / gpt-4o などで最終テキストを整形したい場合に備えたフック。

        いまは llm_manager が None の場合はそのまま返すだけ。
        """
        if self.llm_manager is None:
            # refinement 未使用
            base["refiner_model"] = None
            base["refiner_used"] = False
            base["refiner_status"] = "skipped"
            base["refiner_error"] = ""
            return base

        # ここから下は「本気で refinement したくなったとき」に実装していく。
        # いまは安全のため未使用にしておく。
        base["refiner_model"] = self.refine_model
        base["refiner_used"] = False
        base["refiner_status"] = "skipped"
        base["refiner_error"] = "not implemented"
        return base
