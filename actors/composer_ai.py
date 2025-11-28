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
      - dev_force_model が指定されていれば、そのモデルを強制採用
      - それ以外は JudgeAI3 の chosen_text を優先採用
      - Judge が error の場合は models からフォールバック
      - 将来的な refinement 用に、元テキストや修正フラグをメタ情報として保持
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
              "text": str,                 # 最終テキスト
              "source_model": str,         # 最終テキストの元になったモデル
              "mode": "dev_force" | "judge_choice" | "fallback_from_models",
              "summary": str,

              # ★ デバッグ用追加フィールド
              "base_source_model": str,    # 修正前に採用したモデル名
              "base_text": str,            # 修正前のテキスト
              "is_modified": bool,         # base_text から変更されたかどうか

              # refiner 用メタ
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
                "summary": f"[ComposerAI 0.3] exception: {e}",
                "base_source_model": "",
                "base_text": "",
                "is_modified": False,
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

        # ----------------------------------------
        # 0) 開発用: dev_force_model があれば強制採用
        # ----------------------------------------
        dev_force = str(llm_meta.get("dev_force_model") or "").strip()
        if dev_force:
            info = models.get(dev_force)
            if isinstance(info, dict) and info.get("status") == "ok":
                text = str(info.get("text") or "")
                base = dict(
                    status="ok",
                    text=text,
                    source_model=dev_force,
                    mode="dev_force",
                    summary=f"[ComposerAI 0.3] mode=dev_force, source_model={dev_force}",
                )
                return self._maybe_refine(base=base, llm_meta=llm_meta)
            # 指定モデルが使えない場合は、通常フローへフォールバック

        # ----------------------------------------
        # 1) JudgeAI3 の結果を優先
        # ----------------------------------------
        judge_status = str(judge.get("status", ""))
        chosen_model = str(judge.get("chosen_model") or "")
        chosen_text = str(judge.get("chosen_text") or "")

        if judge_status == "ok" and chosen_model and chosen_text.strip():
            base = dict(
                status="ok",
                text=chosen_text,
                source_model=chosen_model,
                mode="judge_choice",
                summary=(
                    "[ComposerAI 0.3] mode=judge_choice, "
                    f"source_model={chosen_model}, judge_status=ok"
                ),
            )
            return self._maybe_refine(base=base, llm_meta=llm_meta)

        # ----------------------------------------
        # 2) Judge がエラー or 空のときは models からフォールバック
        # ----------------------------------------
        fallback_model, fallback_text = self._fallback_from_models(models)

        if fallback_model and fallback_text.strip():
            base = dict(
                status="ok",
                text=fallback_text,
                source_model=fallback_model,
                mode="fallback_from_models",
                summary=(
                    "[ComposerAI 0.3] mode=fallback_from_models, "
                    f"source_model={fallback_model}, judge_status={judge_status}"
                ),
            )
            return self._maybe_refine(base=base, llm_meta=llm_meta)

        # ----------------------------------------
        # 3) それでも何もない場合
        # ----------------------------------------
        return {
            "status": "error",
            "text": "",
            "source_model": "",
            "mode": "no_text",
            "summary": "[ComposerAI 0.3] no usable text from judge or models",
            "base_source_model": "",
            "base_text": "",
            "is_modified": False,
            "refiner_model": None,
            "refiner_used": False,
            "refiner_status": "skipped",
            "refiner_error": "",
        }

    # -----------------------
    # モデルからのフォールバック
    # -----------------------
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
        ただし「元テキスト」と「修正フラグ」はここで必ず埋める。
        """
        # 「修正前」の情報をまず固定
        base_text = str(base.get("text") or "")
        base_model = str(base.get("source_model") or "")

        base["base_source_model"] = base_model
        base["base_text"] = base_text

        # --- refinement なし運用 ---
        if self.llm_manager is None:
            base["is_modified"] = False
            base["refiner_model"] = None
            base["refiner_used"] = False
            base["refiner_status"] = "skipped"
            base["refiner_error"] = ""
            return base

        # --- refinement を本気で入れたくなったときのための雛形 ---
        # いまは安全のため未使用。
        base["is_modified"] = False
        base["refiner_model"] = self.refine_model
        base["refiner_used"] = False
        base["refiner_status"] = "skipped"
        base["refiner_error"] = "not implemented"
        return base
