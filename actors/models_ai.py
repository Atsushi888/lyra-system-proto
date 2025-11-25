# actors/models_ai.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from llm.llm_manager import LLMManager


class ModelsAI:
    """
    LLMManager を使って複数モデルに問い合わせ、
    各モデルの回答テキスト・usage・エラー状態などを
    まとめて返すユーティリティ。

    - AnswerTalker.run_models() から呼ばれる前提。
    - LLM 本体は LLMRouter → OpenAI / OpenRouter 側に委譲。
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
        LLMRouter / OpenAI SDK から返ってきた値を
        (text, usage) の形に正規化する。

        想定パターン:
        - str
        - (text, usage_dict)
        - OpenAI の ChatCompletion オブジェクト
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
                    # usage が dict 以外のパターンは一旦そのまま詰める
                    usage_dict = {"raw_usage": second}
            if not isinstance(text, str):
                text = "" if text is None else str(text)
            return text, usage_dict

        # すでに str の場合
        if isinstance(raw, str):
            return raw, None

        # OpenAI / OpenRouter の ChatCompletion っぽいオブジェクト
        # choices[0].message.content / usage を拾う
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
                # usage_obj が pydantic / dataclass でも dict 化を試みる
                if isinstance(usage_obj, dict):
                    usage_dict = usage_obj
                else:
                    usage_dict = getattr(usage_obj, "dict", None)
                    if callable(usage_dict):
                        usage_dict = usage_dict()
                    else:
                        usage_dict = {
                            "raw_usage": str(usage_obj),
                        }
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

        # GPT-5.1 の empty-response 問題に対するガード
        if name == "gpt51" and isinstance(text, str) and text.strip() == "":
            # completion_tokens == 0 のときだけ、本当に空レスポンスとみなしてエラー扱い
            comp_tokens = 0
            if isinstance(usage, dict):
                comp_tokens = int(usage.get("completion_tokens", 0) or 0)
            if comp_tokens == 0:
                return {
                    "status": "error",
                    "text": "",
                    "usage": usage,
                    "meta": meta,
                    "error": "empty_response",
                }
            # completion_tokens があるのにテキストだけ取れなかったケースでは、
            # ひとまず空文字のまま「ok」として返す（UI で中身を確認できるようにする）

        if not isinstance(text, str):
            text = "" if text is None else str(text)

        return {
            "status": "ok",
            "text": text,
            "usage": usage,
            "meta": meta,
            "error": None,
        }

        # 通常パターン
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
            現段階では導線のみで、LLM 側での詳細な反映は今後実装予定。

        戻り値フォーマット:
        {
          "gpt4o": { "status": "...", "text": "...", "usage": {...}, "meta": {...}, "error": "..."},
          "gpt51": { ... },
          "hermes": { ... },
          ...
        }
        """
        results: Dict[str, Any] = {}

        if not messages:
            return results

        for name, props in self.model_props.items():
            enabled = props.get("enabled", True)
            if not enabled:
                results[name] = {
                    "status": "disabled",
                    "text": "",
                    "usage": None,
                    "meta": {"mode": mode},
                    "error": "disabled_by_config",
                }
                continue

            try:
                # LLMManager 経由でモデルを実行
                raw = self.llm_manager.call_model(
                    model_name=name,
                    messages=messages,
                    mode=mode,  # ★ 感情モードをパラメータとして渡す（今は導線だけ）
                )
                norm = self._normalize_result(name, raw)
                # meta にも mode を入れておくとデバッグしやすい
                norm_meta = norm.get("meta") or {}
                norm_meta["mode"] = mode
                norm["meta"] = norm_meta
                results[name] = norm
            except Exception as e:
                # 例外はここで吸収して "status=error" として記録
                results[name] = {
                    "status": "error",
                    "text": "",
                    "usage": None,
                    "meta": {"mode": mode},
                    "error": str(e),
                }

        return results
