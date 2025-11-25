# actors/models_ai.py
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
    def _parse_chat_completion_obj(
        obj: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        OpenAI の ChatCompletion っぽいオブジェクトから (text, usage) を抜き出す。
        gpt-5.1 のように message.content が list[part,...] の場合も考慮。
        """
        text = ""
        usage_dict: Optional[Dict[str, Any]] = None

        try:
            choices = getattr(obj, "choices", None) or []
            if choices:
                first = choices[0]
                msg = getattr(first, "message", None)
                content_obj = getattr(msg, "content", "") if msg is not None else ""

                # 1) すでに str
                if isinstance(content_obj, str):
                    text = content_obj

                # 2) list[part, ...]（gpt-5.1 など）
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

                # 3) それ以外は文字列化
                else:
                    text = "" if content_obj is None else str(content_obj)
            else:
                text = ""

            usage_obj = getattr(obj, "usage", None)
            if usage_obj is not None:
                # dict ならそのまま
                if isinstance(usage_obj, dict):
                    usage_dict = usage_obj
                else:
                    # フィールドを手作業で拾う
                    prompt_tokens = getattr(usage_obj, "prompt_tokens", None)
                    completion_tokens = getattr(usage_obj, "completion_tokens", None)
                    total_tokens = getattr(usage_obj, "total_tokens", None)

                    tmp: Dict[str, Any] = {}
                    if prompt_tokens is not None:
                        tmp["prompt_tokens"] = prompt_tokens
                    if completion_tokens is not None:
                        tmp["completion_tokens"] = completion_tokens
                    if total_tokens is not None:
                        tmp["total_tokens"] = total_tokens

                    if tmp:
                        usage_dict = tmp
                    else:
                        dmethod = getattr(usage_obj, "dict", None)
                        if callable(dmethod):
                            usage_dict = dmethod()
                        else:
                            usage_dict = {"raw_usage": str(usage_obj)}

        except Exception:
            # どうしてもパースできなければ空文字＋ usage なし
            text = ""
            usage_dict = None

        return text, usage_dict

    @classmethod
    def _extract_text_and_usage(cls, raw: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        LLMAdapter / OpenAI SDK などから返ってきた値を
        (text, usage) の形に正規化する。

        想定パターン:
        - str
        - (text, usage_dict)
        - (ChatCompletion, usage_dict)
        - ChatCompletion オブジェクト
        - それ以外 → 文字列化して usage=None
        """

        # 1) (text, usage) 形式
        if isinstance(raw, tuple):
            if not raw:
                return "", None

            text = raw[0]
            usage_dict: Optional[Dict[str, Any]] = None

            if len(raw) >= 2:
                second = raw[1]
                if isinstance(second, dict):
                    usage_dict = second
                else:
                    usage_dict = {"raw_usage": second}

            # 先頭が ChatCompletion オブジェクトの場合は、さらに中身を剥がす
            if not isinstance(text, str) and hasattr(text, "choices"):
                cc_text, cc_usage = cls._parse_chat_completion_obj(text)
                text = cc_text
                if cc_usage:
                    # usage が両方ある場合はマージ
                    if isinstance(usage_dict, dict):
                        merged = dict(usage_dict)
                        merged.update(cc_usage)
                        usage_dict = merged
                    else:
                        usage_dict = cc_usage

            if not isinstance(text, str):
                text = "" if text is None else str(text)

            return text, usage_dict

        # 2) すでに str の場合
        if isinstance(raw, str):
            return raw, None

        # 3) ChatCompletion っぽいオブジェクト（旧ルート / デバッグ用フォールバック）
        if hasattr(raw, "choices"):
            return cls._parse_chat_completion_obj(raw)

        # 4) それ以外はとりあえず文字列化
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
