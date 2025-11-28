# actors/composer_ai.py
from __future__ import annotations

from typing import Any, Dict, Optional

from llm.llm_manager import LLMManager


class ComposerAI:
    """
    llm_meta["models"] と llm_meta["judge"] をもとに、
    「最終返答テキスト」を組み立てるクラス。

    v0.3 の方針:
      - 原則として追加で LLM を呼ばない（安定性優先）
      - JudgeAI3 の chosen_text を優先採用
      - Judge が error の場合は models からフォールバック
      - llm_meta["dev_force_model"] が指定されていれば、そのモデルを強制採用
      - 将来的な refinement 用のフックだけ用意しておく
      - ベース案と最終案の両方を llm_meta['composer'] に保存して、
        デバッグ画面から見比べられるようにする
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
            いまは None を指定して、LLM 呼び出しなし運用でも OK。

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
              "mode": "judge_choice" | "fallback_from_models"
                      | "dev_force" | "exception" など,
              "base_text": str,          # Judge / fallback による元テキスト
              "base_source_model": str,  # そのときのモデル名
              "dev_force_model": str,    # 強制指定されたモデル名（なければ空）
              "is_modified": bool,       # base_text から変化したか
              "summary": str,
              "refiner_model": Optional[str],
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
                "base_text": "",
                "base_source_model": "",
                "dev_force_model": str(llm_meta.get("dev_force_model") or ""),
                "is_modified": False,
                "summary": f"[ComposerAI 0.3] exception: {e}",
                "refiner_model": None,
                "refiner_used": False,
                "refiner_status": "error",
                "refiner_error": str(e),
            }

    # =======================
    # 内部実装
    # =======================
    def _safe_compose(self, llm_meta: Dict[str, Any]) -> Dict[str, Any]:
        models = llm_meta.get("models") or {}
        judge = llm_meta.get("judge") or {}

        dev_force_model = str(llm_meta.get("dev_force_model") or "").strip() or ""

        # 1) Judge / fallback から「ベース案」を決める
        base_source_model, base_text, base_mode = self._select_base(models, judge)

        if not base_source_model or not base_text.strip():
            # 何も使えるテキストがなければエラー
            return {
                "status": "error",
                "text": "",
                "source_model": "",
                "mode": "no_text",
                "base_text": "",
                "base_source_model": "",
                "dev_force_model": dev_force_model,
                "is_modified": False,
                "summary": "[ComposerAI 0.3] no usable text from judge or models",
                "refiner_model": None,
                "refiner_used": False,
                "refiner_status": "skipped",
                "refiner_error": "",
            }

        # 2) dev_force_model が指定されていれば、それを優先採用してみる
        final_source_model, final_text, final_mode = self._apply_dev_force(
            models=models,
            base_source_model=base_source_model,
            base_text=base_text,
            base_mode=base_mode,
            dev_force_model=dev_force_model,
        )

        # 3) 結果をまとめて、必要なら refinement フックを通す
        is_modified = (final_text != base_text) or (
            final_source_model != base_source_model
        )

        result = {
            "status": "ok",
            "text": final_text,
            "source_model": final_source_model,
            "mode": final_mode,
            "base_text": base_text,
            "base_source_model": base_source_model,
            "dev_force_model": dev_force_model,
            "is_modified": is_modified,
            "summary": (
                f"[ComposerAI 0.3] mode={final_mode}, "
                f"source_model={final_source_model}, "
                f"base_model={base_source_model}, "
                f"dev_force_model={dev_force_model or '-'}"
            ),
        }

        return self._maybe_refine(result, llm_meta)

    # -----------------------
    # Judge / models からベース案を決める
    # -----------------------
    def _select_base(
        self,
        models: Dict[str, Any],
        judge: Dict[str, Any],
    ) -> tuple[str, str, str]:
        """
        まず JudgeAI3 の結果を尊重し、ダメなら models からフォールバックする。
        戻り値: (source_model, text, mode)
        """
        judge_status = str(judge.get("status", ""))
        chosen_model = str(judge.get("chosen_model") or "")
        chosen_text = str(judge.get("chosen_text") or "")

        # 1) JudgeAI3 の採用結果を優先
        if judge_status == "ok" and chosen_model and chosen_text.strip():
            return chosen_model, chosen_text, "judge_choice"

        # 2) Judge がエラー or 空のときは models からフォールバック
        fallback_model, fallback_text = self._fallback_from_models(models)
        if fallback_model and fallback_text.strip():
            return fallback_model, fallback_text, "fallback_from_models"

        # 3) それでも何もない場合
        return "", "", "no_text"

    # -----------------------
    # dev_force_model の適用
    # -----------------------
    def _apply_dev_force(
        self,
        *,
        models: Dict[str, Any],
        base_source_model: str,
        base_text: str,
        base_mode: str,
        dev_force_model: str,
    ) -> tuple[str, str, str]:
        """
        llm_meta["dev_force_model"] が指定されていれば、そのモデルの出力で
        ベース案を上書きする。

        失敗した場合（該当モデルがない／status!=ok／text 空）は、
        ベース案のまま返す。
        """
        if not dev_force_model:
            return base_source_model, base_text, base_mode

        info = models.get(dev_force_model)
        if not isinstance(info, Dict):
            # 指定されたモデル情報が存在しない
            return base_source_model, base_text, base_mode

        if info.get("status") != "ok":
            # モデル呼び出しに失敗している
            return base_source_model, base_text, base_mode

        text = str(info.get("text") or "").strip()
        if not text:
            # テキストが空なので使えない
            return base_source_model, base_text, base_mode

        # 正常に取得できたので dev_force 採用
        return dev_force_model, text, "dev_force"

    # -----------------------
    # モデルからのフォールバック
    # -----------------------
    @staticmethod
    def _fallback_from_models(models: Dict[str, Any]) -> tuple[str, str]:
        """
        models から、もっとも無難なテキストを 1 つ選ぶ。

        いまは「status=ok かつ text が空でないモデル」を
        gpt51 → grok → gemini の優先度で探し、
        見つからなければ最初の ok を返す。
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
    # （任意）refinement フック
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
