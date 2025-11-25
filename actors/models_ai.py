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
        self.llm_manager = llm_manager
        self.model_props: Dict[str, Dict[str, Any]] = (
            self.llm_manager.get_model_props()
        )

    # ============================
    # 内部ヘルパ
    # ============================
    @staticmethod
    def _extract_text_and_usage(raw: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        LLMRouter などから返ってきた値を (text, usage) に正規化。

        想定パターン:
        - str
        - (text, usage_dict)
          * text が list[ContentPart] になる gpt-5.1 もここで処理する
        - ChatCompletion オブジェクト（旧ルート用の保険）
        - それ以外 → 文字列化して usage=None
        """

        # (text, usage) 形式
        if isinstance(raw, tuple):
            if not raw:
                return "", None

            content_obj = raw[0]
            usage_dict: Optional[Dict[str, Any]] = None

            if len(raw) >= 2:
                second = raw[1]
                if isinstance(second, dict):
                    usage_dict = second
                else:
                    usage_dict = {"raw_usage": second}

            # --- text 部分を型ごとに整形 ---
            # 1) すでに str
            if isinstance(content_obj, str):
                text = content_obj

            # 2) gpt-5.1 などで list[part, part, ...] になっている場合
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

            # 3) その他は文字列化
            else:
                text = "" if content_obj is None else str(content_obj)

            return text, usage_dict

        # すでに str の場合
        if isinstance(raw, str):
            return raw, None

        # ChatCompletion っぽいオブジェクト（保険）
        try:
            choices = getattr(raw, "choices", None)
            if choices and isinstance(choices, list) and choices:
                first = choices[0]
                msg = getattr(first, "message", None)
                content_obj = getattr(msg, "content", None)
            else:
                content_obj = ""

            # text 整形（上と同じロジック）
            if isinstance(content_obj, str):
                content = content_obj
            elif isinstance(content_obj, list):
                parts: List[str] = []
                for p in content_obj:
                    t = getattr(p, "text", None)
                    if t is None and isinstance(p, dict):
                        t = p.get("text")
                    if t is None:
                        t = str(p)
                    parts.append(t)
                content = "".join(parts)
            else:
                content = "" if content_obj is None else str(content_obj)

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
            return content, usage_dict
        except Exception:
            return str(raw), None

    def _normalize_result(self, name: str, raw: Any) -> Dict[str, Any]:
        """
        各モデルの生返却値 raw を統一フォーマットに整形する。
        """
        text, usage = self._extract_text_and_usage(raw)
        meta: Dict[str, Any] = {}

        # GPT-5.1 の「完全な空返答」を検出してエラー扱いにする
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
        """
        results: Dict[str, Any] = {}

        if not messages:
            return results

        mode_key = (mode or "normal").lower()

        for name, props in self.model_props.items():
            enabled_cfg = props.get("enabled", True)

            # 1) gpt4o は常時不参加
            if name == "gpt4o":
                enabled = False
            # 2) Hermes（旧）は erotic のときだけ参加
            elif name == "hermes":
                enabled = enabled_cfg and (mode_key == "erotic")
            # 3) それ以外は設定どおり
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

            try:
                raw = self.llm_manager.call_model(
                    model_name=name,
                    messages=messages,
                    mode=mode_key,  # LLMManager 側で pop して記録だけ
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
