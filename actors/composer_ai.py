# actors/composer_ai.py
from __future__ import annotations

from typing import Any, Dict, Optional

from llm.llm_manager import LLMManager


class ComposerAI:
    """
    llm_meta["models"] / llm_meta["judge"] / llm_meta["style_hint"]
    さらに llm_meta["world_state"] / llm_meta["scene_emotion"]
    をもとに、「最終返答テキスト」を組み立てるクラス。

    v0.4 方針（Lyra-System 専用最適化）:
      - 原則: Judge が選んだ本文に「世界観に基づく微調整」を行い最終テキスト決定
      - LLM を追加で呼ばずに 90% 完了できるよう最適化
      - ただし llm_manager が存在する場合、Refiner を任意で発動可能
      - dev_force_model があれば最優先
      - world_state（場所・時刻・距離・同伴状態）を軽く整形ロジックに反映
      - reply_length_mode（short/normal/long/story）は
        主に system_prompt と Refiner へのヒントとして利用
    """

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        refine_model: str = "gpt51",
    ) -> None:
        self.llm_manager = llm_manager
        self.refine_model = refine_model

    # ============================================================
    # 公開 API
    # ============================================================
    def compose(self, llm_meta: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self._safe_compose(llm_meta)
        except Exception as e:
            dev_force_model = str(llm_meta.get("dev_force_model") or "")
            return {
                "status": "error",
                "text": "",
                "source_model": "",
                "mode": "exception",
                "summary": f"[ComposerAI 0.4] exception: {e}",
                "base_text": "",
                "base_source_model": "",
                "dev_force_model": dev_force_model,
                "is_modified": False,
                "refiner_model": self.refine_model if self.llm_manager else None,
                "refiner_used": False,
                "refiner_status": "error",
                "refiner_error": str(e),
            }

    # ============================================================
    # 内部実装
    # ============================================================
    def _safe_compose(self, llm_meta: Dict[str, Any]) -> Dict[str, Any]:
        models: Dict[str, Any] = llm_meta.get("models") or {}
        judge: Dict[str, Any] = llm_meta.get("judge") or {}
        world_state: Dict[str, Any] = llm_meta.get("world_state") or {}
        scene_emotion: Dict[str, Any] = llm_meta.get("scene_emotion") or {}
        reply_length_mode: str = str(llm_meta.get("reply_length_mode") or "auto")

        dev_force_model: str = str(llm_meta.get("dev_force_model") or "").strip()

        # ---------------------------------------------------------
        # 1) dev_force_model を最優先
        # ---------------------------------------------------------
        base_model, base_text, mode = self._select_base_text_dev_force(
            models=models,
            dev_force_model=dev_force_model,
        )

        # ---------------------------------------------------------
        # 2) dev_force が無ければ Judge の選択
        # ---------------------------------------------------------
        if not base_model:
            jm, jt, m = self._select_base_text_from_judge(models=models, judge=judge)
            base_model, base_text, mode = jm, jt, m

        # ---------------------------------------------------------
        # 3) それでも駄目なら models 内からフォールバック
        # ---------------------------------------------------------
        if not base_model:
            fm, ft = self._fallback_from_models(models)
            base_model, base_text = fm, ft
            mode = "fallback_from_models"

        # ---------------------------------------------------------
        # 4) 完全に取得失敗
        # ---------------------------------------------------------
        if not base_model or not base_text.strip():
            return {
                "status": "error",
                "text": "",
                "source_model": "",
                "mode": "no_text",
                "summary": "[ComposerAI 0.4] no usable text",
                "base_text": "",
                "base_source_model": "",
                "dev_force_model": dev_force_model,
                "is_modified": False,
                "refiner_model": self.refine_model if self.llm_manager else None,
                "refiner_used": False,
                "refiner_status": "skipped",
                "refiner_error": "",
            }

        # ---------------------------------------------------------
        # 5) world_state を踏まえた「軽い整形」
        # ---------------------------------------------------------
        composed = self._inject_world_context(base_text, world_state)

        # ---------------------------------------------------------
        # 6) Refiner を噛ませる（任意）
        # ---------------------------------------------------------
        result = {
            "status": "ok",
            "text": composed,
            "source_model": base_model,
            "mode": mode,
            "summary": f"[ComposerAI 0.4] mode={mode}, source={base_model}",
            "base_text": composed,
            "base_source_model": base_model,
            "dev_force_model": dev_force_model,
            "refiner_model": self.refine_model if self.llm_manager else None,
            "is_modified": False,
            "refiner_used": False,
            "refiner_status": "pending" if self.llm_manager else "skipped",
            "refiner_error": "",
            "reply_length_mode": reply_length_mode,
        }

        return self._maybe_refine(result, llm_meta)

    # ============================================================
    # world_state 情報を文章に反映
    # ============================================================
    def _inject_world_context(self, text: str, world_state: Dict[str, Any]) -> str:
        """
        SceneAI によって作成された world_state には以下が含まれる:

            world_state = {
                "location_name": "駅前",
                "time_of_day": "朝",
                "time_str": "07:20",
                "weather": "clear",
                "party_mode": "both" / "alone"
            }

        ここでは LLM を使わず、文章最終行に only-one-line の軽い注釈を加え、
        「状況に合った発話に調整」だけを行う。

        LLM に丸投げせず、誤改変が起きない安全な方式。
        """

        if not isinstance(world_state, dict):
            return text

        loc = world_state.get("location_name")
        tod = world_state.get("time_of_day")
        tstr = world_state.get("time_str")
        weather = world_state.get("weather")
        pmode = world_state.get("party_mode")

        addon_parts = []

        if loc:
            addon_parts.append(f"（現在地: {loc}）")
        if tod:
            addon_parts.append(f"（時間帯: {tod}）")
        if tstr:
            addon_parts.append(f"（時刻: {tstr}）")
        if weather:
            addon_parts.append(f"（天候: {weather}）")
        if pmode == "alone":
            addon_parts.append("（現在、あなたは一人きりだ）")

        if not addon_parts:
            return text

        # 文章末尾に「※状況補足」を一行だけ付与する。
        extra = " ".join(addon_parts)
        return text.rstrip() + "\n\n" + extra

    # ============================================================
    # dev_force_model 選択
    # ============================================================
    @staticmethod
    def _select_base_text_dev_force(
        *,
        models: Dict[str, Any],
        dev_force_model: str,
    ) -> tuple[str, str, str]:
        if not dev_force_model:
            return "", "", ""

        info = models.get(dev_force_model)
        if not isinstance(info, Dict):
            return "", "", ""

        if info.get("status") != "ok":
            return "", "", ""

        text = str(info.get("text") or "").strip()
        if not text:
            return "", "", ""

        return dev_force_model, text, "dev_force"

    # ============================================================
    # Judge 選択
    # ============================================================
    @staticmethod
    def _select_base_text_from_judge(
        *,
        models: Dict[str, Any],
        judge: Dict[str, Any],
    ) -> tuple[str, str, str]:
        judge_status = str(judge.get("status", ""))
        chosen_model = str(judge.get("chosen_model") or "")
        chosen_text = str(judge.get("chosen_text") or "")

        if judge_status == "ok" and chosen_model and chosen_text.strip():
            return chosen_model, chosen_text, "judge_choice"

        if chosen_model and isinstance(models.get(chosen_model), Dict):
            info = models[chosen_model]
            if info.get("status") == "ok":
                text = str(info.get("text") or "").strip()
                if text:
                    return chosen_model, text, "judge_model_fallback"

        return "", "", ""

    # ============================================================
    # fallback（modelsから拾う）
    # ============================================================
    @staticmethod
    def _fallback_from_models(models: Dict[str, Any]) -> tuple[str, str]:
        preferred_order = ["gpt51", "gpt4o", "hermes", "grok", "gemini"]

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

    # ============================================================
    # Refiner 呼び出し
    # ============================================================
    def _maybe_refine(
        self,
        base: Dict[str, Any],
        llm_meta: Dict[str, Any],
    ) -> Dict[str, Any]:

        if self.llm_manager is None:
            base["refiner_used"] = False
            base["refiner_status"] = "skipped"
            base["refiner_error"] = ""
            return base

        try:
            refined = self._call_refiner(
                text=str(base.get("text") or ""),
                llm_meta=llm_meta,
            )
        except Exception as e:
            base["refiner_used"] = False
            base["refiner_status"] = "error"
            base["refiner_error"] = str(e)
            return base

        if not refined or refined.strip() == str(base.get("base_text", "")).strip():
            base["refiner_used"] = False
            base["refiner_status"] = "ok_empty"
            base["is_modified"] = False
            return base

        base["text"] = refined
        base["refiner_used"] = True
        base["refiner_status"] = "ok"
        base["is_modified"] = True
        return base

    # ============================================================
    # 実際の Refiner LLM 呼び出し
    # ============================================================
    def _call_refiner(self, text: str, llm_meta: Dict[str, Any]) -> str:
        if not self.llm_manager:
            raise RuntimeError("llm_manager is None")

        style_hint = str(llm_meta.get("style_hint") or "")
        length_mode = str(llm_meta.get("reply_length_mode") or "auto").lower()

        # 基本ポリシー
        system_prompt = (
            "あなたは日本語の文章スタイリストです。\n"
            "意味や内容を変えず、文体を整えて自然で読みやすい日本語にしてください。\n"
        )

        # 長さモード別のヒント
        if length_mode == "short":
            system_prompt += (
                "文章量は可能な限りコンパクトに保ち、2〜3文程度に収まるよう意識してください。\n"
                "不要な言い回しや重複を削り、簡潔にまとめてください。\n"
            )
        elif length_mode in ("long", "story"):
            system_prompt += (
                "文章量は元のテキストと同程度か、少しだけ厚くして構いませんが、"
                "分量が元の文章のおよそ2倍を超えないようにしてください。\n"
                "情景や感情のニュアンスを自然な範囲で補っても構いません。\n"
            )
        else:
            system_prompt += (
                "文章量を大きく変えないでください。長さはだいたい元のテキストと同程度に保ってください。\n"
            )

        if style_hint:
            system_prompt += (
                "\n以下は参考にすべき文体のメモです（内容改変は厳禁）：\n"
                f"{style_hint}\n"
            )

        user_msg = (
            "次のテキストを文体だけ整えてください。\n"
            "本文のみを返し、説明や注釈は不要です。\n\n"
            "-----\n"
            f"{text}\n"
            "-----"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        mgr = self.llm_manager

        if hasattr(mgr, "chat"):
            response = mgr.chat(
                model=self.refine_model,
                messages=messages,
                temperature=0.5,
                max_tokens=900,
            )
        else:
            response = mgr.chat_completion(
                model=self.refine_model,
                messages=messages,
                temperature=0.5,
                max_tokens=900,
            )

        refined = None

        if isinstance(response, str):
            refined = response
        elif isinstance(response, tuple):
            refined = str(response[0])
        elif isinstance(response, dict):
            refined = (
                response.get("text")
                or response.get("content")
            )
            if refined is None and "choices" in response:
                try:
                    refined = response["choices"][0]["message"]["content"]
                except Exception:
                    refined = None

        if not refined:
            raise RuntimeError("Refiner response has no text")

        return str(refined).strip()
