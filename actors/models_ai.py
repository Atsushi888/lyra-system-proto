from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from llm.llm_manager import LLMManager


class ModelsAI:
    """
    LLMManager を使って複数モデルに問い合わせ、
    各モデルの回答テキスト・usage・エラー状態などを
    まとめて返すユーティリティ。
    """

    def __init__(self, llm_manager: LLMManager) -> None:
        # LLMManager インスタンスを保持
        self.llm_manager = llm_manager
        # モデル定義（vendor / router_fn / priority / enabled / extra ...）
        self.model_props: Dict[str, Dict[str, Any]] = (
            self.llm_manager.get_model_props()
        )

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
          （gpt-5.1 のように message.content が list[part] でも対応）
        - dict（OpenRouter / Grok / Gemini など）
        - それ以外 → 文字列化して usage=None
        """

        # ----------------------------------------
        # (text, usage) 形式
        # ----------------------------------------
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

        # ----------------------------------------
        # すでに str の場合
        # ----------------------------------------
        if isinstance(raw, str):
            return raw, None

        # ----------------------------------------
        # dict 形式（OpenRouter / Grok / Gemini など）
        # ----------------------------------------
        if isinstance(raw, dict):
            try:
                choices = raw.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content", "")
                    # gpt-5.1 型の list[part] にも対応
                    if isinstance(content, list):
                        parts: List[str] = []
                        for p in content:
                            t = None
                            if isinstance(p, dict):
                                t = p.get("text")
                            if t is None and hasattr(p, "text"):
                                t = getattr(p, "text", None)
                            if t is None:
                                t = str(p)
                            parts.append(t)
                        text = "".join(parts)
                    else:
                        text = content or ""
                else:
                    text = ""
                usage_dict = raw.get("usage")
                return text, usage_dict
            except Exception:
                # 失敗したらとりあえず文字列化
                return str(raw), None

        # ----------------------------------------
        # ChatCompletion っぽいオブジェクト（OpenAI）
        # ----------------------------------------
        try:
            choices = getattr(raw, "choices", None)
            if choices and isinstance(choices, list) and choices:
                first = choices[0]
                msg = getattr(first, "message", None)
                content = getattr(msg, "content", "")

                # ★ gpt-5.1: content が list[part] のケース
                if isinstance(content, list):
                    parts: List[str] = []
                    for p in content:
                        t = getattr(p, "text", None)
                        if t is None and isinstance(p, dict):
                            t = p.get("text")
                        if t is None:
                            t = str(p)
                        parts.append(t)
                    text = "".join(parts)
                else:
                    text = content or ""
            else:
                text = ""

            usage_obj = getattr(raw, "usage", None)
            usage_dict: Optional[Dict[str, Any]] = None
            if usage_obj is not None:
                if isinstance(usage_obj, dict):
                    usage_dict = usage_obj
                else:
                    tmp = getattr(usage_obj, "dict", None)
                    if callable(tmp):
                        usage_dict = tmp()
                    else:
                        usage_dict = {"raw_usage": str(usage_obj)}

            return text, usage_dict
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
        すべての有効モデルに対して LLM 呼び出しを行い、結果を dict で返す。

        mode:
            "normal" / "erotic" / "debate" など、
            現在の Judge / Emotion モードを示す。
        """
        results: Dict[str, Any] = {}

        if not messages:
            return results

        mode_key = (mode or "normal").lower()

        for name, props in self.model_props.items():
            # ==== 参加可否フィルタ ====
            enabled_cfg = props.get("enabled", True)

            # 1) gpt4o は常時不参加
            if name == "gpt4o":
                enabled = False

            # 2) Hermes（旧）は erotic のときだけ参加
            elif name == "hermes":
                enabled = enabled_cfg and (mode_key == "erotic")

            # 3) それ以外（gpt51 / grok / gemini / hermes_new）は config どおり
            else:
                enabled = enabled_cfg

            if not enabled:
                results[name] = {
                    "status": "disabled",
                    "text": "",
                    "usage": None,
                    "meta": {"mode": mode_key},
                    "error": "disabled_by_config",
                }
                continue

            # ==== 実行 ====
            try:
                raw = self.llm_manager.call_model(
                    model_name=name,
                    messages=messages,
                    mode=mode_key,
                )
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
