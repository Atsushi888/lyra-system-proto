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
    def _parse_openai_like_completion(obj: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        ChatCompletion っぽいオブジェクトから text / usage を抜き出す。
        gpt-5.1 のように content が list[part] の場合にも対応。
        """
        text = ""
        usage_dict: Optional[Dict[str, Any]] = None

        try:
            choices = getattr(obj, "choices", None) or []
            if choices:
                first = choices[0]
                msg = getattr(first, "message", None)
                content_obj = getattr(msg, "content", "") or ""

                if isinstance(content_obj, str):
                    text = content_obj
                elif isinstance(content_obj, list):
                    parts: List[str] = []
                    for p in content_obj:
                        t = getattr(p, "text", None)
                        if t is None and isinstance(p, dict):
                            t = p.get("text")
                        if t is None:
                            t = str(p)
                        parts.append(t)
                    text = "".join(parts)
                else:
                    text = str(content_obj)
            else:
                text = ""

            usage_obj = getattr(obj, "usage", None)
            if usage_obj is not None:
                if isinstance(usage_obj, dict):
                    usage_dict = usage_obj
                else:
                    usage_dict = {
                        "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                        "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
                        "total_tokens": getattr(usage_obj, "total_tokens", 0),
                    }
        except Exception:
            text = str(obj)
            usage_dict = None

        return text, usage_dict

    @staticmethod
    def _looks_like_completion(obj: Any) -> bool:
        return hasattr(obj, "choices")

    @classmethod
    def _extract_text_and_usage(cls, raw: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        LLMAdapter / OpenAI SDK などから返ってきた値を
        (text, usage) の形に正規化する。

        想定パターン:
        - str
        - (text, usage_dict)
        - (ChatCompletion, usage_dict)
        - ChatCompletion 単体
        - それ以外 → 文字列化して usage=None
        """
        # (text, usage) 形式 or (completion, usage) 形式
        if isinstance(raw, tuple):
            if not raw:
                return "", None

            head = raw[0]
            tail = raw[1] if len(raw) >= 2 else None

            # 0番目が ChatCompletion っぽい場合はそこから text を抜く
            if cls._looks_like_completion(head):
                text, usage_from_head = cls._parse_openai_like_completion(head)

                if isinstance(tail, dict):
                    usage_dict = tail
                else:
                    usage_dict = usage_from_head

                return text, usage_dict

            # ふつうの (text, usage) 形式
            text = head
            usage_dict: Optional[Dict[str, Any]] = None
            if isinstance(tail, dict):
                usage_dict = tail
            elif tail is not None:
                usage_dict = {"raw_usage": tail}
            if not isinstance(text, str):
                text = "" if text is None else str(text)
            return text, usage_dict

        # すでに str の場合
        if isinstance(raw, str):
            return raw, None

        # ChatCompletion 単体
        if cls._looks_like_completion(raw):
            return cls._parse_openai_like_completion(raw)

        # それ以外：保険として文字列化
        return str(raw), None

    def _normalize_result(self, name: str, raw: Any) -> Dict[str, Any]:
        """
        各モデルの生返却値 raw を統一フォーマットに整形する。
        """
        text, usage = self._extract_text_and_usage(raw)
        meta: Dict[str, Any] = {}

        # GPT-5.1 の empty-response ガード
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
