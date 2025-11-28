# actors/composer_ai.py
from __future__ import annotations

from typing import Any, Dict, Optional

from llm.llm_manager import LLMManager


class ComposerAI:
    """
    llm_meta["models"] と llm_meta["judge"] をもとに、
    「最終返答テキスト」を組み立てるクラス。

    v0.4 方針:
      - dev_force_model があればそれを最優先で採用
      - なければ JudgeAI3 の chosen を優先し、ダメなら models からフォールバック
      - 任意で refiner LLM（例: gpt51）にかけてフローリア口調で整形
      - 仕上げ前後のテキスト両方を llm_meta['composer'] に残す
    """

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        refine_model: str = "gpt51",
    ) -> None:
        self.llm_manager = llm_manager
        self.refine_model = refine_model

    # =======================
    # 公開 API
    # =======================
    def compose(self, llm_meta: Dict[str, Any]) -> Dict[str, Any]:
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
                "refiner_status": "error",
                "refiner_error": str(e),
            }

    # =======================
    # 内部実装
    # =======================
    def _safe_compose(self, llm_meta: Dict[str, Any]) -> Dict[str, Any]:
        models: Dict[str, Any] = llm_meta.get("models") or {}
        judge: Dict[str, Any] = llm_meta.get("judge") or {}

        dev_force_model = str(llm_meta.get("dev_force_model") or "").strip()

        # ======================================================
        # 1) dev_force_model があれば、最優先でそのモデルを採用
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
                            f"[ComposerAI 0.4] mode=dev_force, "
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
        base_model, base_text, base_mode = self._select_base(models, judge)

        if base_model and base_text.strip():
            base = {
                "status": "ok",
                "text": base_text,
                "source_model": base_model,
                "mode": base_mode,
                "summary": (
                    f"[ComposerAI 0.4] mode={base_mode}, "
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
            "summary": "[ComposerAI 0.4] no usable text from judge or models",
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
        chosen_model = str(judge.get("chosen_model") or "").strip()
        chosen_text = str(judge.get("chosen_text") or "")

        if judge_status == "ok" and chosen_model:
            info = models.get(chosen_model) or {}
            model_text = str(info.get("text") or "").strip()
            text = model_text or chosen_text
            if text.strip():
                return chosen_model, text, "judge_choice"

        fb_model, fb_text = self._fallback_from_models(models)
        if fb_model and fb_text.strip():
            return fb_model, fb_text, "fallback_from_models"

        return "", "", "no_text"

    # -----------------------
    # モデルからのフォールバック
    # -----------------------
    @staticmethod
    def _fallback_from_models(models: Dict[str, Any]) -> tuple[str, str]:
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
    # （任意）refinement 本体
    # -----------------------
    def _maybe_refine(
        self,
        base: Dict[str, Any],
        llm_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        base["text"] をもとに、refine_model で最終仕上げを行う。

        - llm_manager が None の場合は何もしない
        - 何らかの理由で refine に失敗したら、元テキストのまま返す
        """
        if self.llm_manager is None:
            base["refiner_model"] = None
            base["refiner_used"] = False
            base["refiner_status"] = "skipped"
            base["refiner_error"] = ""
            return base

        ref_model = self.refine_model or "gpt51"
        original_text = str(base.get("text") or "")
        if not original_text.strip():
            base["refiner_model"] = ref_model
            base["refiner_used"] = False
            base["refiner_status"] = "skipped"
            base["refiner_error"] = "empty_base_text"
            return base

        style_hint = str(llm_meta.get("composer_style_hint") or "").strip()
        emotion = llm_meta.get("emotion") or {}
        mode = emotion.get("mode", "normal")

        emo_desc = (
            f"現在の感情モードは「{mode}」です。"
            f"好意={emotion.get('affection', 0.0):.2f}, "
            f"性的な高ぶり={emotion.get('arousal', 0.0):.2f}, "
            f"緊張={emotion.get('tension', 0.0):.2f}, "
            f"悲しみ={emotion.get('sadness', 0.0):.2f}, "
            f"ワクワク={emotion.get('excitement', 0.0):.2f} です。"
        )

        system_parts = [
            "あなたは『フローリア』としてロールプレイする日本語話者のライティングアシスタントです。",
            "与えられたテキストを、フローリアの一人称・口調を保ったまま、読みやすく自然な日本語に整えてください。",
            "情報を削り過ぎないようにしつつ、冗長な部分があればさりげなく整理してください。",
            "見出しや箇条書き、装飾記号は使わず、純粋な文章だけで返答してください。",
        ]
        if style_hint:
            system_parts.append(
                "以下のスタイル指示も厳守してください:\n" + style_hint
            )
        system_parts.append("数値や内部パラメータの説明は一切出力しないでください。")
        system_parts.append(emo_desc)

        system_prompt = "\n\n".join(system_parts)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "次のテキストを、上記のルールに従って丁寧に整えてください。\n\n"
                    "【入力テキスト】\n"
                    f"{original_text}\n\n"
                    "【出力フォーマット】\n"
                    "・フローリアとしての一人称・口調のまま\n"
                    "・自然な日本語の文章のみ\n"
                    "・不要な説明やメタなコメントは付けないこと"
                ),
            },
        ]

        try:
            # ★ LLMManager の実装に合わせてここだけ調整してね
            result = self.llm_manager.chat(
                model_name=ref_model,
                messages=messages,
                temperature=0.7,
                max_tokens=900,
            )
            if isinstance(result, tuple):
                refined_text, usage = result
            else:
                refined_text = str(result)
                usage = {}

            refined_text = str(refined_text or "").strip()

            base["refiner_model"] = ref_model
            base["refiner_used"] = True
            base["refiner_status"] = "ok"
            base["refiner_error"] = ""

            if not refined_text:
                base["refiner_status"] = "ok_empty"
                return base

            if refined_text != original_text:
                base["text"] = refined_text
                base["is_modified"] = True

            base["refiner_usage"] = usage
            return base

        except Exception as e:
            base["refiner_model"] = ref_model
            base["refiner_used"] = False
            base["refiner_status"] = "error"
            base["refiner_error"] = str(e)
            return base
