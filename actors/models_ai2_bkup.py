# actors/models_ai2.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from actors.llm_ai import LLMAIManager, create_default_llm_ai_manager


class ModelsAI2:
    """
    LLMAIManager を使って複数モデルに問い合わせ、
    各モデルの回答テキスト・usage・エラー状態などを
    まとめて返すユーティリティ（新実装版）。

    - AnswerTalker.run_models() から呼ばれる前提。
    - 実際のベンダー呼び出しは actors/llm_adapters/ 配下の LLMAI サブクラスに委譲。
    """

    def __init__(self, llm_ai_manager: Optional[LLMAIManager] = None) -> None:
        # LLMAIManager インスタンスを保持
        self.llm_ai_manager: LLMAIManager = llm_ai_manager or create_default_llm_ai_manager()

    # ============================
    # 内部ヘルパ
    # ============================
    @staticmethod
    def _extract_text_and_usage(raw: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        (text, usage_dict) 形式 or その他から (text, usage) を取り出す。
        """
        # すでに (text, usage) 形式
        if isinstance(raw, tuple):
            if not raw:
                return "", None
            text = raw[0]
            usage_dict = None
            if len(raw) >= 2:
                second = raw[1]
                if isinstance(second, dict):
                    usage_dict = second
                else:
                    usage_dict = {"raw_usage": second}
            if not isinstance(text, str):
                text = "" if text is None else str(text)
            return text, usage_dict

        # str の場合
        if isinstance(raw, str):
            return raw, None

        # それ以外はとりあえず文字列化
        return str(raw), None

    def _normalize_result(self, name: str, raw: Any) -> Dict[str, Any]:
        """
        各モデルの生返却値 raw を統一フォーマットに整形する。
        """
        text, usage = self._extract_text_and_usage(raw)
        meta: Dict[str, Any] = {}

        # GPT-5.1 の「完全な空返答」をエラー扱いにするガード
        comp_tokens = 0
        if isinstance(usage, dict):
            try:
                comp_tokens = int(usage.get("completion_tokens", 0) or 0)
            except Exception:
                comp_tokens = 0

        if (
            name == "gpt51"
            and isinstance(text, str)
            and text.strip() == ""
            and comp_tokens == 0
        ):
            return {
                "status": "error",
                "text": "",
                "usage": usage,
                "meta": meta,
                "error": "empty_response",
            }

        if not isinstance(text, str):
            text = "" if text is None else str(text)

        return {
            "status": "ok",
            "text": text,
            "usage": usage,
            "meta": meta,
            "error": None,
        }

    # ============================
    # 公開 API
    # ============================
    def collect(
        self,
        messages: List[Dict[str, str]],
        mode: str = "normal",
    ) -> Dict[str, Any]:
        """
        すべての LLMAI サブクラスに対して LLM 呼び出しを行い、結果を dict で返す。

        戻り値フォーマット（既存 ModelsAI と互換）:
        {
          "gpt51": { "status": "...", "text": "...", "usage": {...}, "meta": {...}, "error": "..."},
          "grok":  { ... },
          "gemini":{ ... },
          "hermes":{ ... },
          ...
        }
        """
        results: Dict[str, Any] = {}

        if not messages:
            return results

        mode_key = (mode or "normal").lower()

        for name, model in self.llm_ai_manager.all_models().items():
            # 参加可否判定（enabled + judge_mode フラグ）
            enabled = model.should_answer(mode_key)

            if not enabled:
                results[name] = {
                    "status": "disabled",
                    "text": "",
                    "usage": None,
                    "meta": {"mode": mode_key},
                    "error": "disabled_by_config_or_mode",
                }
                continue

            # ==== 実行 ====
            try:
                raw = model.call(messages=messages)
                norm = self._normalize_result(name, raw)
                norm_meta = norm.get("meta") or {}
                norm_meta["mode"] = mode_key
                norm["meta"] = norm_meta
                results[name] = norm
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "text": "",
                    "usage": None,
                    "meta": {"mode": mode_key},
                    "error": str(e),
                }

        return results
