# actors/composer_ai.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable, Tuple


# refiner 用の関数シグネチャ
RefinerFn = Callable[
    [
        str,                      # question_text: ユーザーの最後の質問
        str,                      # source_model: Judge が採用したモデル名 (例: "gpt51")
        str,                      # chosen_text: そのモデルの回答テキスト
        List[Dict[str, Any]],     # other_candidates: 落選候補
        Dict[str, Any],           # llm_meta: 全メタ情報
        Optional[str],            # refiner_model: 仕上げに使う LLM 名
    ],
    str,                          # refined_text
]


class ComposerAI:
    """
    AI回答パイプラインの「仕上げ」担当クラス。

    役割:
      - ModelsAI.collect が集めた各モデル出力 (llm_meta["models"])
      - JudgeAI2.process が選んだ採択情報 (llm_meta["judge"])
        を入力として、「最終返答テキスト」を決定する。

    v0.2.0 のポイント:
      - 基本は Judge の chosen_text をベースとする
      - refiner（任意のコールバック）を通じて、別LLM/GPT-5.1 などで
        「他の候補も踏まえた仕上げ（リファイン）」を行える
      - どの LLM を仕上げに使ったかを llm_meta['composer'] に記録する
    """

    def __init__(
        self,
        *,
        refiner: Optional[RefinerFn] = None,
        refiner_model: Optional[str] = None,
        enable_refine: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        refiner:
            実際に LLM を呼んで仕上げを行うコールバック。
            None の場合はリファインを行わない（Judge の選択をそのまま採用）。

        refiner_model:
            refiner が使うべき LLM 名（例: "gpt51"）。
            コールバック側でこれを見て router_fn を切り替えることを想定。

        enable_refine:
            False にすると、refiner が渡されていてもリファインをスキップする。
        """
        self.refiner = refiner
        self.refiner_model = refiner_model
        self.enable_refine = enable_refine

        self.name = "ComposerAI"
        self.version = "0.2.0"

    # ============================
    # 公開インターフェース
    # ============================
    def compose(self, llm_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        AnswerTalker から呼び出されるメインメソッド。

        Parameters
        ----------
        llm_meta:
            AnswerTalker が管理しているメタ情報辞書。
            期待しているキー:
              - "models": ModelsAI.collect(...) の結果
              - "judge" : JudgeAI2.process(...) の結果
              - "messages": Persona.build_messages() で組んだ messages（任意）

        Returns
        -------
        Dict[str, Any]
            例:
            {
              "status": "ok",
              "text": "...最終返答テキスト...",
              "source_model": "gpt51",
              "mode": "judge_pass_through+refined",
              "summary": "...デバッグ用サマリ...",
              "composer": {"name": "ComposerAI", "version": "0.2.0"},
              "refiner": {
                  "model": "gpt51",
                  "used": True,
                  "status": "ok",
                  "error": "",
              },
            }
        """
        if not isinstance(llm_meta, dict):
            return {
                "status": "error",
                "error": "llm_meta must be dict[str, Any]",
                "text": "",
            }

        models = llm_meta.get("models", {})
        judge = llm_meta.get("judge", {})

        if not isinstance(models, dict):
            models = {}
        if not isinstance(judge, dict):
            judge = {}

        # 1) Judge の結果を読み取る
        judge_status = judge.get("status")
        if judge_status == "ok":
            chosen_model = str(judge.get("chosen_model") or "")
            chosen_text = str(judge.get("chosen_text") or "")
            reason = str(judge.get("reason") or "")
            reason_text = str(judge.get("reason_text") or "")
            candidates = judge.get("candidates") or []
        else:
            chosen_model = ""
            chosen_text = ""
            reason = ""
            reason_text = ""
            candidates = []

        # 2) ベースとなるテキスト / モデルを決める
        final_text = ""
        source_model = ""

        if judge_status == "ok" and chosen_text:
            final_text = chosen_text
            source_model = chosen_model or self._fallback_model_name(models)
            mode = "judge_pass_through"
        else:
            # Judge がエラー or 全員スコア0 の場合は models から決める
            final_text, source_model = self._fallback_from_models(models)
            mode = "fallback_from_models"

        # 3) リファイン（仕上げ）処理
        refiner_meta: Dict[str, Any] = {
            "model": self.refiner_model,
            "used": False,
            "status": "skipped",  # "ok" / "skipped" / "error" / "ok_empty"
            "error": "",
        }

        if (
            self.enable_refine
            and self.refiner is not None
            and isinstance(final_text, str)
            and final_text.strip()
        ):
            try:
                messages = llm_meta.get("messages") or []
                question = self._extract_last_user_content(messages)

                other_candidates = self._build_other_candidates(
                    candidates=candidates,
                    source_model=source_model,
                )

                refined = self.refiner(
                    question,
                    source_model,
                    final_text,
                    other_candidates,
                    llm_meta,
                    self.refiner_model,
                )

                refiner_meta["used"] = True

                if isinstance(refined, str) and refined.strip():
                    final_text = refined
                    mode = mode + "+refined"
                    refiner_meta["status"] = "ok"
                else:
                    refiner_meta["status"] = "ok_empty"
            except Exception as e:
                refiner_meta["used"] = True
                refiner_meta["status"] = "error"
                refiner_meta["error"] = str(e)
                mode = mode + "+refine_error"

        # 4) ポストプロセス（テキスト装飾など）
        final_text = self._postprocess_text(
            text=final_text,
            source_model=source_model,
            models=models,
            judge=judge,
        )

        # 5) サマリ文字列（デバッグ用）
        summary = self._build_summary(
            source_model=source_model,
            mode=mode,
            judge_status=judge_status,
            judge_reason=reason,
            judge_reason_text=reason_text,
            models=models,
            refiner_meta=refiner_meta,
        )

        return {
            "status": "ok",
            "text": final_text,
            "source_model": source_model,
            "mode": mode,
            "summary": summary,
            "composer": {
                "name": self.name,
                "version": self.version,
            },
            "refiner": refiner_meta,
        }

    # ============================
    # 内部: 候補処理
    # ============================

    def _fallback_model_name(self, models: Dict[str, Dict[str, Any]]) -> str:
        """
        chosen_model が空だった場合などに、models から
        それっぽい model 名を一つ選ぶための補助ヘルパー。
        """
        if not isinstance(models, dict):
            return ""

        for name, info in models.items():
            if isinstance(info, dict) and info.get("status") == "ok":
                return str(name)

        for name in models.keys():
            return str(name)

        return ""

    def _fallback_from_models(
        self,
        models: Dict[str, Dict[str, Any]],
    ) -> Tuple[str, str]:
        """
        JudgeAI2 がエラーのときに models からテキストを選ぶ。
        """
        if not isinstance(models, dict):
            return "", ""

        for name, info in models.items():
            if not isinstance(info, dict):
                continue
            if info.get("status") != "ok":
                continue
            text = info.get("text") or ""
            if text:
                return str(text), str(name)

        for name, info in models.items():
            if not isinstance(info, dict):
                continue
            text = info.get("text") or ""
            if text:
                return str(text), str(name)

        return "", ""

    def _build_other_candidates(
        self,
        *,
        candidates: Any,
        source_model: str,
    ) -> List[Dict[str, Any]]:
        """
        Judge の candidates から「落選候補」だけを抽出して返す。
        """
        result: List[Dict[str, Any]] = []
        if not isinstance(candidates, list):
            return result

        for c in candidates:
            if not isinstance(c, dict):
                continue
            name = c.get("name") or c.get("model")
            if not name or name == source_model:
                continue
            result.append(c)

        return result

    # ============================
    # 内部: ポストプロセス
    # ============================

    def _postprocess_text(
        self,
        *,
        text: str,
        source_model: str,
        models: Dict[str, Dict[str, Any]],
        judge: Dict[str, Any],
    ) -> str:
        """
        最終テキストに対して、必要なら後処理を行う。
        現バージョンでは「何もしない」。
        """
        if not text:
            return ""
        return text

    # ============================
    # 内部: サマリ & 補助
    # ============================

    def _build_summary(
        self,
        *,
        source_model: str,
        mode: str,
        judge_status: Optional[str],
        judge_reason: str,
        judge_reason_text: str,
        models: Dict[str, Dict[str, Any]],
        refiner_meta: Dict[str, Any],
    ) -> str:
        """
        Streamlit サイドで「どのモデルがどうだったか」を
        一瞥できるようにするための簡易サマリ文字列。
        """
        lines: List[str] = []

        lines.append(f"[ComposerAI {self.version}] mode={mode}")
        lines.append(f"source_model={source_model or 'N/A'}")
        lines.append(f"judge_status={judge_status or 'N/A'}")

        if judge_reason:
            lines.append(f"judge_reason={judge_reason}")
        if judge_reason_text:
            lines.append(f"judge_reason_text={judge_reason_text}")

        model = refiner_meta.get("model") or "N/A"
        lines.append(
            f"refiner: model={model}, "
            f"used={refiner_meta.get('used')}, "
            f"status={refiner_meta.get('status')}, "
            f"error={refiner_meta.get('error', '')}"
        )

        if isinstance(models, dict) and models:
            lines.append("models:")
            for name, info in models.items():
                if not isinstance(info, dict):
                    lines.append(f"  - {name}: (invalid info)")
                    continue
                status = info.get("status", "unknown")
                text = info.get("text") or ""
                length = len(text)
                lines.append(f"  - {name}: status={status}, len={length}")
        else:
            lines.append("models: <empty>")

        return "\n".join(lines)

    @staticmethod
    def _extract_last_user_content(messages: List[Dict[str, Any]]) -> str:
        """
        messages から最後の user メッセージの content を抽出。
        見つからなければ空文字。
        """
        if not isinstance(messages, list):
            return ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
        return ""
