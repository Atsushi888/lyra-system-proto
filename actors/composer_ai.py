# actors/composer_ai.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from llm.llm_manager import LLMManager


class ComposerAI:
    """
    llm_meta["models"] と llm_meta["judge"] をもとに、
    「最終返答テキスト」を組み立てるクラス。

    v0.3 方針:
      - 原則として追加で LLM を呼ばない（安定性優先）
      - JudgeAI3 の chosen_text を「基本線」としつつ、
        scene_prefs（シーン設定）に応じて他モデルを選び直すことを許可
      - 開発用に dev_force_model でモデル固定も可能
      - 将来的に llm_manager を渡して refinement を有効化できる設計
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
              "mode":
                  "dev_force"
                | "judge_choice"
                | "fallback_from_models"
                | "composer_override"
                | "no_text"
                | "exception",
              "summary": str,
              ...
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
        models: Dict[str, Any] = llm_meta.get("models") or {}
        judge: Dict[str, Any] = llm_meta.get("judge") or {}

        # --------------------------------------------------
        # 0) 開発用: dev_force_model があればそのモデルを優先採用
        # --------------------------------------------------
        dev_force = str(llm_meta.get("dev_force_model") or "").strip()
        if dev_force and dev_force in models:
            info = models.get(dev_force) or {}
            if info.get("status") == "ok":
                text = str(info.get("text") or "").strip()
                if text:
                    base = {
                        "status": "ok",
                        "text": text,
                        "source_model": dev_force,
                        "mode": "dev_force",
                        "summary": (
                            "[ComposerAI 0.3] mode=dev_force, "
                            f"source_model={dev_force}"
                        ),
                    }
                    return self._maybe_refine(base=base, llm_meta=llm_meta)

        # --------------------------------------------------
        # 1) JudgeAI3 の結果から「ベース候補」を決定
        # --------------------------------------------------
        judge_status = str(judge.get("status", ""))
        judge_model = str(judge.get("chosen_model") or "")
        judge_text = str(judge.get("chosen_text") or "")

        base_mode = "judge_choice"
        base_model: str
        base_text: str

        if judge_status == "ok" and judge_model and judge_text.strip():
            base_model = judge_model
            base_text = judge_text
        else:
            # Judge がエラー or 空のときは models からフォールバック
            fb_model, fb_text = self._fallback_from_models(models)
            if not fb_model or not fb_text.strip():
                return {
                    "status": "error",
                    "text": "",
                    "source_model": "",
                    "mode": "no_text",
                    "summary": (
                        "[ComposerAI 0.3] no usable text from judge or models "
                        f"(judge_status={judge_status})"
                    ),
                }
            base_model = fb_model
            base_text = fb_text
            base_mode = "fallback_from_models"

        # --------------------------------------------------
        # 2) scene_prefs があれば「短い案を優先」などのスコアリング
        # --------------------------------------------------
        scene_prefs: Dict[str, Any] = llm_meta.get("scene_prefs") or {}
        chosen_model, chosen_text, scoring_info = self._select_with_scene_prefs(
            models=models,
            base_model=base_model,
            base_text=base_text,
            scene_prefs=scene_prefs,
        )

        if chosen_model == base_model:
            mode = base_mode
        else:
            mode = "composer_override"

        summary_parts = [
            f"[ComposerAI 0.3] mode={mode}",
            f"base_model={base_model}",
            f"chosen_model={chosen_model}",
            f"judge_status={judge_status}",
        ]
        if scoring_info:
            summary_parts.append(f"scoring={scoring_info}")

        base = {
            "status": "ok",
            "text": chosen_text,
            "source_model": chosen_model,
            "mode": mode,
            "summary": ", ".join(summary_parts),
        }
        return self._maybe_refine(base=base, llm_meta=llm_meta)

    # -----------------------
    # モデルからのフォールバック
    # -----------------------
    @staticmethod
    def _fallback_from_models(models: Dict[str, Any]) -> Tuple[str, str]:
        """
        models から、もっとも無難なテキストを 1 つ選ぶ。

        いまは「status=ok かつ text が空でないモデル」を
        好みの順（gpt51 / grok / gemini）で探し、それ以外は
        最初に見つかった ok モデルを返す。
        """
        if not isinstance(models, dict):
            return "", ""

        preferred_order = ["gpt51", "grok", "gemini"]

        for name in preferred_order:
            info = models.get(name)
            if not isinstance(info, dict):
                continue
            if info.get("status") != "ok":
                continue
            text = str(info.get("text") or "").strip()
            if text:
                return name, text

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
    # シーン設定に基づくスコアリング
    # -----------------------
    def _select_with_scene_prefs(
        self,
        *,
        models: Dict[str, Any],
        base_model: str,
        base_text: str,
        scene_prefs: Dict[str, Any],
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        scene_prefs が指定されていれば、各モデル案をスコアリングして
        採用モデルを決める。

        scene_prefs 例:
            {
                "prefer_short": True,
                "max_chars": 220,
                "weight_short": 0.6,
            }
        """
        if not scene_prefs or not isinstance(models, dict):
            return base_model, base_text, {}

        prefer_short = bool(scene_prefs.get("prefer_short", False))
        max_chars = int(scene_prefs.get("max_chars", 0) or 0)
        if max_chars <= 0:
            max_chars = 300
        weight_short = float(scene_prefs.get("weight_short", 0.6))

        candidates = []
        debug_scores: Dict[str, Dict[str, Any]] = {}

        for name, info in models.items():
            if not isinstance(info, dict):
                continue
            if info.get("status") != "ok":
                continue

            text = str(info.get("text") or "").strip()
            if not text:
                continue

            length = len(text)
            score = 0.0

            if name == base_model:
                score += 1.0

            if prefer_short:
                ratio = min(1.0, max(0.0, length / max_chars))
                score += weight_short * (1.0 - ratio)

                if length > max_chars:
                    over_ratio = min(2.0, (length - max_chars) / max_chars)
                    score -= weight_short * 0.5 * over_ratio

            candidates.append((score, name, text))
            debug_scores[name] = {
                "len": length,
                "score": round(score, 3),
                "is_base": name == base_model,
            }

        if not candidates:
            return base_model, base_text, {}

        def sort_key(item: Any) -> Any:
            s, n, _ = item
            is_base = (n == base_model)
            return (s, is_base)

        best_score, best_name, best_text = max(candidates, key=sort_key)

        debug_info = {
            "prefer_short": prefer_short,
            "max_chars": max_chars,
            "weight_short": weight_short,
            "best_score": round(best_score, 3),
            "scores": debug_scores,
        }

        return best_name, best_text, debug_info

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

        base["refiner_model"] = self.refine_model
        base["refiner_used"] = False
        base["refiner_status"] = "skipped"
        base["refiner_error"] = "not implemented"
        return base
