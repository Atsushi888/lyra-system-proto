# actors/composer_ai.py
from __future__ import annotations

from typing import Any, Dict, Optional

from llm.llm_manager import LLMManager


class ComposerAI:
    """
    llm_meta["models"] と llm_meta["judge"] をもとに、
    「最終返答テキスト」を組み立てるクラス。

    v0.3 方針:
      - 原則として追加で LLM を呼ばない（安定性優先）
      - dev_force_model が指定されていれば、それを最優先で採用
      - それ以外は JudgeAI の chosen を優先し、ダメなら models からフォールバック
      - 将来的な refinement 用に llm_manager フックのみ用意
    """

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        refine_model: str = "gpt4o",
    ) -> None:
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
              "mode": "dev_force" | "judge_choice" | "fallback_from_models",
              "summary": str,
              "base_text": str,
              "base_source_model": str,
              "dev_force_model": str | "",
              "is_modified": bool,
              "refiner_model": str | None,
              "refiner_used": bool,
              "refiner_status": str,
              "refiner_error": str,
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
                "base_text": "",
                "base_source_model": "",
                "dev_force_model": str(llm_meta.get("dev_force_model") or ""),
                "is_modified": False,
                "refiner_model": None,
                "refiner_used": False,
                "refiner_status": "skipped",
                "refiner_error": "",
            }

    # =======================
    # 内部実装
    # =======================
    def _safe_compose(self, llm_meta: Dict[str, Any]) -> Dict[str, Any]:
        models: Dict[str, Any] = llm_meta.get("models") or {}
        judge: Dict[str, Any] = llm_meta.get("judge") or {}

        dev_force_model = str(llm_meta.get("dev_force_model") or "").strip()

        # ======================================================
        # 1) dev_force_model があれば、それを最優先で採用
        # ======================================================
        if dev_force_model:
            info = models.get(dev_force_model)
            if isinstance(info, dict) and info.get("status") == "ok":
                dev_text = str(info.get("text") or "")
                if dev_text.strip():
                    base = {
                        "status": "ok",
                        "text": dev_text,
                        "source_model": dev_force_model,
                        "mode": "dev_force",
                        "summary": (
                            f"[ComposerAI 0.3] mode=dev_force, "
                            f"source_model={dev_force_model}"
                        ),
                        "base_text": dev_text,
                        "base_source_model": dev_force_model,
                        "dev_force_model": dev_force_model,
                        "is_modified": False,
                    }
                    return self._maybe_refine(base, llm_meta)

        # ======================================================
        # 2) 通常ルート: Judge → models フォールバック
        # ======================================================
        base_model, base_text = self._select_base_text(models, judge)

        if base_model and base_text.strip():
            base = {
                "status": "ok",
                "text": base_text,
                "source_model": base_model,
                "mode": "judge_choice"
                if judge.get("status") == "ok"
                else "fallback_from_models",
                "summary": (
                    f"[ComposerAI 0.3] mode="
                    f"{'judge_choice' if judge.get('status') == 'ok' else 'fallback_from_models'}, "
                    f"source_model={base_model}, judge_status={judge.get('status')}"
                ),
                "base_text": base_text,
                "base_source_model": base_model,
                "dev_force_model": dev_force_model,
                "is_modified": False,
            }
            return self._maybe_refine(base, llm_meta)

        # ======================================================
        # 3) それでも何もない場合
        # ======================================================
        return {
            "status": "error",
            "text": "",
            "source_model": "",
            "mode": "no_text",
            "summary": "[ComposerAI 0.3] no usable text from judge or models",
            "base_text": "",
            "base_source_model": "",
            "dev_force_model": dev_force_model,
            "is_modified": False,
            "refiner_model": None,
            "refiner_used": False,
            "refiner_status": "skipped",
            "refiner_error": "",
        }

    # -----------------------
    # Judge / models からのベース選択
    # -----------------------
    def _select_base_text(
        self,
        models: Dict[str, Any],
        judge: Dict[str, Any],
    ) -> tuple[str, str]:
        """
        まず Judge の chosen を見て、ダメなら models からフォールバック。
        """

        judge_status = str(judge.get("status", ""))
        chosen_model = str(judge.get("chosen_model") or "").strip()
        chosen_text = str(judge.get("chosen_text") or "")

        # 1) Judge が正常で chosen_text があればそれを採用
        if judge_status == "ok" and chosen_model:
            # models 側のテキストがあるならそちらを優先、なければ chosen_text
            info = models.get(chosen_model) or {}
            model_text = str(info.get("text") or "").strip()
            text = model_text or chosen_text
            if text.strip():
                return chosen_model, text

        # 2) ダメなら models からフォールバック
        return self._fallback_from_models(models)

    @staticmethod
    def _fallback_from_models(models: Dict[str, Any]) -> tuple[str, str]:
        """
        models から、もっとも無難なテキストを 1 つ選ぶ。

        いまは「status=ok かつ text が空でないモデル」を
        gpt51 → grok → gemini の優先順で探し、それでも無ければ
        最初に見つかった ok モデルを返す。
        """
        if not isinstance(models, dict):
            return "", ""

        preferred_order = ["gpt51", "grok", "gemini"]

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
