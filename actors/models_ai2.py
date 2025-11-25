# actors/models_ai2.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from actors.llm_ai import LLMAIRegistry


class ModelsAI2:
    """
    LLMAIRegistry を使って複数モデルに問い合わせ、
    各モデルの回答テキスト・usage・エラー状態などを
    まとめて返すユーティリティ（新実装版）。
    """

    def __init__(
        self,
        *,
        llm_manager: Any = None,              # 将来拡張用（今は未使用）
        registry: Optional[LLMAIRegistry] = None,
    ) -> None:
        self.llm_manager = llm_manager
        self.registry: LLMAIRegistry = registry or LLMAIRegistry.create_default()

    # ============================
    # 内部ヘルパ
    # ============================
    @staticmethod
    def _extract_text_and_usage(raw: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        LLMAdapter / OpenAI SDK などから返ってきた値を
        (text, usage) の形に正規化する。

        想定パターン:
        - str
        - (text, usage_dict)
        - OpenAI の ChatCompletion オブジェクト
        - dict(OpenRouter/Grok など)
        - それ以外 → 文字列化して usage=None
        """
        # (text, usage) 形式
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

        # すでに str の場合
        if isinstance(raw, str):
            return raw, None

        # dict(OpenRouter/Grok など) っぽい場合
        if isinstance(raw, dict):
            try:
                choices = raw.get("choices") or []
                text = ""
                if choices:
                    msg = choices[0].get("message") or {}
                    text = msg.get("content", "") or ""
                usage = raw.get("usage")
                if not isinstance(text, str):
                    text = "" if text is None else str(text)
                if isinstance(usage, dict):
                    return text, usage
                elif usage is not None:
                    return text, {"raw_usage": usage}
                else:
                    return text, None
            except Exception:
                return str(raw), None

        # ChatCompletion っぽいオブジェクト
        try:
            choices = getattr(raw, "choices", None)
            if choices and isinstance(choices, list) and choices:
                first = choices[0]
                msg = getattr(first, "message", None)
                content = getattr(msg, "content", None)
                if not isinstance(content, str):
                    content = "" if content is None else str(content)
            else:
                content = ""

            usage_obj = getattr(raw, "usage", None)
            usage_dict = None
            if usage_obj is not None:
                if isinstance(usage_obj, dict):
                    usage_dict = usage_obj
                else:
                    tmp = getattr(usage_obj, "dict", None)
                    if callable(tmp):
                        usage_dict = tmp()
                    else:
                        usage_dict = {"raw_usage": str(usage_obj)}
            return content, usage_dict
        except Exception:
            # 失敗したらとりあえず文字列化
            return str(raw), None

    def _normalize_result(self, name: str, raw: Any) -> Dict[str, Any]:
        """
        各モデルの生返却値 raw を統一フォーマットに整形する。
        """
        text, usage = self._extract_text_and_usage(raw)
        meta: Dict[str, Any] = {}

        # GPT-5.1 の empty-response ガード
        # usage.completion_tokens も 0（あるいは usage 自体が無い）場合だけ
        # 「完全な空返答」とみなしてエラー扱いにする。
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
        すべてのレジストリ登録モデルに対して LLM 呼び出しを行い、結果を dict で返す。

        mode:
            "normal" / "erotic" / "debate" など、
            現在の Judge / Emotion モードを示す。
        """
        results: Dict[str, Any] = {}

        if not messages:
            return results

        mode_key = (mode or "normal").lower()

        for name, ai in self.registry.all().items():
            # ==== 参加可否フィルタ ====
            if (not ai.enabled) or (not ai.should_answer(mode_key)):
                results[name] = {
                    "status": "disabled",
                    "text": "",
                    "usage": None,
                    "meta": {"mode": mode_key},
                    "error": "disabled_by_config_or_mode",
                }
                continue

            # API キーが無い場合は config で無効扱い
            has_key = True
            try:
                has_key = ai.has_api_key()
            except Exception:
                has_key = False

            if not has_key:
                results[name] = {
                    "status": "disabled",
                    "text": "",
                    "usage": None,
                    "meta": {"mode": mode_key},
                    "error": "no_api_key",
                }
                continue

            # ==== 実行 ====
            try:
                raw = ai.call(messages=messages)
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
