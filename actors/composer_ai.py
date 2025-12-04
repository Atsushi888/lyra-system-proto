# actors/composer_ai.py
from __future__ import annotations

from typing import Any, Dict, Optional

from llm.llm_manager import LLMManager


class ComposerAI:
    """
    llm_meta["models"] / llm_meta["judge"] / llm_meta["style_hint"] などをもとに、
    「最終返答テキスト」を組み立てるクラス。

    v0.3 + world_state refine 方針:
      - 原則として追加で LLM を呼ばずに完結
      - ただし llm_manager が与えられていれば Refiner を任意で起動可能
      - dev_force_model があればそれを最優先で採用（開発・検証用）
      - Refiner には world_state / scene_emotion を渡し、
        シーンと矛盾しないように“軽く整形”だけ行う
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
        """
        llm_meta 全体を受け取り、最終返答テキストとメタ情報を返す。
        """
        try:
            return self._safe_compose(llm_meta)
        except Exception as e:
            dev_force_model = str(llm_meta.get("dev_force_model") or "")
            return {
                "status": "error",
                "text": "",
                "source_model": "",
                "mode": "exception",
                "summary": f"[ComposerAI 0.3] exception: {e}",
                "base_text": "",
                "base_source_model": "",
                "dev_force_model": dev_force_model,
                "is_modified": False,
                "refiner_model": self.refine_model if self.llm_manager else None,
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
        dev_force_model: str = str(llm_meta.get("dev_force_model") or "").strip()

        # 1) dev_force_model が指定されていれば最優先
        base_model, base_text, mode = self._select_base_text_dev_force(
            models=models,
            dev_force_model=dev_force_model,
        )

        # 2) dev_force が使えなかった場合は JudgeAI の結果を優先
        if not base_model:
            jm, jt, m = self._select_base_text_from_judge(models=models, judge=judge)
            base_model, base_text, mode = jm, jt, m

        # 3) それでも決まらなければ models からフォールバック
        if not base_model:
            fm, ft = self._fallback_from_models(models)
            base_model, base_text = fm, ft
            mode = "fallback_from_models"

        # 4) まだ駄目なら完全エラー
        if not base_model or not base_text.strip():
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
                "refiner_model": self.refine_model if self.llm_manager else None,
                "refiner_used": False,
                "refiner_status": "skipped",
                "refiner_error": "",
            }

        # -----------------------
        # ここまでで「元テキスト」が確定
        # -----------------------
        summary = (
            f"[ComposerAI 0.3] mode={mode}, "
            f"source_model={base_model}, dev_force_model={dev_force_model or '-'}"
        )

        result: Dict[str, Any] = {
            "status": "ok",
            "text": base_text,
            "source_model": base_model,
            "mode": mode,
            "summary": summary,
            # Refiner まわり
            "base_text": base_text,
            "base_source_model": base_model,
            "dev_force_model": dev_force_model,
            "is_modified": False,
            "refiner_model": self.refine_model if self.llm_manager else None,
            "refiner_used": False,
            "refiner_status": "skipped" if self.llm_manager is None else "pending",
            "refiner_error": "",
        }

        # 必要なら Refiner を噛ませる
        return self._maybe_refine(result, llm_meta)

    # -----------------------
    # dev_force_model の処理
    # -----------------------
    @staticmethod
    def _select_base_text_dev_force(
        *,
        models: Dict[str, Any],
        dev_force_model: str,
    ) -> tuple[str, str, str]:
        """
        dev_force_model が有効なら、それをベースとして採用する。
        """
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

    # -----------------------
    # JudgeAI からの選択
    # -----------------------
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

        # Judge がダメでも、chosen_model が models 側に居れば拾ってみる
        if chosen_model and isinstance(models.get(chosen_model), Dict):
            info = models[chosen_model]
            if info.get("status") == "ok":
                text = str(info.get("text") or "").strip()
                if text:
                    return chosen_model, text, "judge_model_fallback"

        return "", "", ""

    # -----------------------
    # モデルからのフォールバック
    # -----------------------
    @staticmethod
    def _fallback_from_models(models: Dict[str, Any]) -> tuple[str, str]:
        """
        models から、もっとも無難なテキストを 1 つ選ぶ。
        """
        if not isinstance(models, dict):
            return "", ""

        # 優先順位（好み）
        preferred_order = ["gpt51", "gpt4o", "hermes", "grok", "gemini"]

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
        将来的に gpt-5.1 などで最終テキストを整形したい場合に備えたフック。
        world_state / scene_emotion を見て、シーンと矛盾しない範囲で“軽く”整形。
        """
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
            base["refiner_error"] = ""
            base["is_modified"] = False
            return base

        base["text"] = refined
        base["refiner_used"] = True
        base["refiner_status"] = "ok"
        base["refiner_error"] = ""
        base["is_modified"] = True
        return base

    # -----------------------
    # 実際の Refiner 呼び出し
    # -----------------------
    def _call_refiner(self, text: str, llm_meta: Dict[str, Any]) -> str:
        """
        llm_manager を使って、最終テキストを「軽く整形」する。

        - style_hint が llm_meta にあれば、それも踏まえて整える。
        - world_state / scene_emotion があれば、シーンと矛盾しないように注意させる。
        - それでも「内容を大きく書き換えない」ことを最優先。
        """
        if not self.llm_manager:
            raise RuntimeError("llm_manager is None")

        style_hint = str(llm_meta.get("style_hint") or "")

        # world_state の軽い要約をつくる
        ws = llm_meta.get("world_state") or {}
        locs = ws.get("locations", {})
        time = ws.get("time", {})
        party = ws.get("party", {})

        player_loc = locs.get("player", "")
        floria_loc = locs.get("floria", "")
        slot = time.get("slot", "")
        time_str = time.get("time_str", "")
        party_mode = party.get("mode", "")

        scene_lines = []
        if player_loc:
            scene_lines.append(f"プレイヤーの現在地: {player_loc}")
        if floria_loc:
            scene_lines.append(f"フローリアの現在地: {floria_loc}")
        if slot or time_str:
            scene_lines.append(f"時間帯スロット: {slot} / 時刻: {time_str}")
        if party_mode:
            scene_lines.append(f"パーティ状態: {party_mode}")

        scene_summary = "\n".join(scene_lines) if scene_lines else "（特筆すべきシーン情報なし）"

        system_prompt = (
            "あなたは日本語の文章スタイリスト兼コンシステンシー・チェック担当です。\n"
            "与えられたテキストを、意味や内容を変えずに、読みやすく自然な文体に整えてください。\n"
            "文章量（長さ）は大きく変えないでください。\n"
            "また、以下のシーン情報と矛盾しないように注意してください。\n"
            "矛盾がある場合は、最小限の書き換えで整合性をとりますが、"
            "物語の大枠や重要な出来事を変えてはいけません。\n\n"
            "【シーン情報（参考）】\n"
            f"{scene_summary}\n"
        )

        if style_hint:
            system_prompt += (
                "\n以下は、望ましい文体のメモです。文体のみを参考にし、"
                "内容や設定を改変しないでください：\n"
                f"{style_hint}\n"
            )

        user_msg = (
            "次のテキストを、意味を変えずに、文体だけ整えてください。\n"
            "ただし、上記のシーン情報（現在地や時間帯）と大きく矛盾しないようにしてください。\n"
            "出力はテキスト本文のみを返し、解説や前置きは書かないでください。\n\n"
            "----- テキストここから -----\n"
            f"{text}\n"
            "----- テキストここまで -----"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        mgr = self.llm_manager

        # LLMManager の実装に応じて呼び分け
        response: Any
        if hasattr(mgr, "chat"):
            response = mgr.chat(
                model=self.refine_model,
                messages=messages,
                temperature=0.7,
                max_tokens=900,
            )
        elif hasattr(mgr, "chat_completion"):
            response = mgr.chat_completion(
                model=self.refine_model,
                messages=messages,
                temperature=0.7,
                max_tokens=900,
            )
        else:
            raise RuntimeError(
                "LLMManager に 'chat' / 'chat_completion' のどちらも実装されていません。"
            )

        refined_text: Optional[str] = None

        if isinstance(response, str):
            refined_text = response
        elif isinstance(response, tuple):
            refined_text = str(response[0])
        elif isinstance(response, dict):
            refined_text = response.get("text") or response.get("content")
            if refined_text is None and "choices" in response:
                try:
                    refined_text = response["choices"][0]["message"]["content"]
                except Exception:
                    refined_text = None

        if not refined_text:
            raise RuntimeError("Refiner response has no text content")

        return str(refined_text).strip()
