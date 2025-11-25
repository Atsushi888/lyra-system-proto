# actors/models_ai2.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from actors.llm_ai import LLMAIRegistry, LLMAI


class ModelsAI2:
    """
    LLMAI（gpt51 / grok / gemini / hermes / hermes_new / gpt4o など）の
    レジストリを使って、複数モデルに問い合わせるユーティリティ。

    - AnswerTalker.run_models() から呼ばれることを想定。
    - ここでは LLMManager / 旧 router には触らず、
      各 LLMAI が内包している adapter / router に丸投げする。
    """

    def __init__(
        self,
        llm_manager: Optional[Any] = None,
        registry: Optional[LLMAIRegistry] = None,
    ) -> None:
        # 将来的な互換用として llm_manager を受け取るが、ここでは使わない
        self.llm_manager = llm_manager

        # LLMAI のレジストリ（デフォルト: gpt51 / grok / gemini / hermes / hermes_new / gpt4o）
        self.registry: LLMAIRegistry = registry or LLMAIRegistry.create_default()

        # name -> LLMAI のキャッシュ
        self.models: Dict[str, LLMAI] = self.registry.all()

    # ============================
    # 内部ヘルパ
    # ============================
    @staticmethod
    def _normalize_gpt51_guard(
        name: str,
        text: str,
        usage: Optional[Dict[str, Any]],
    ) -> Tuple[bool, str]:
        """
        gpt51 の「空レス問題」を検出するためのガード。
        - text が空
        - usage.completion_tokens が 0 or 無し
        の場合のみ True を返して error 扱いにする。
        """
        if name != "gpt51":
            return False, text

        if not isinstance(text, str):
            text = "" if text is None else str(text)

        comp_tokens = 0
        if isinstance(usage, dict):
            try:
                comp_tokens = int(usage.get("completion_tokens", 0) or 0)
            except Exception:
                comp_tokens = 0

        if text.strip() == "" and comp_tokens == 0:
            # 完全な空返答とみなす
            return True, ""
        return False, text

    # ============================
    # 公開 API
    # ============================
    def collect(
        self,
        messages: List[Dict[str, str]],
        mode: str = "normal",
    ) -> Dict[str, Any]:
        """
        すべての LLMAI に対して LLM 呼び出しを行い、結果を dict で返す。

        mode:
            "normal" / "erotic" / "debate" など、現在の Judge モード。

        戻り値フォーマット（llm_meta["models"] に格納される形）:
        {
          "gpt51": {
            "status": "ok" | "error" | "disabled",
            "text": str,
            "usage": Optional[Dict[str, Any]],
            "meta": { "mode": "normal", ... },
            "error": Optional[str],
          },
          "hermes": { ... },
          "grok": { ... },
          ...
        }
        """
        results: Dict[str, Any] = {}

        if not messages:
            return results

        mode_key = (mode or "normal").lower()

        for name, ai in self.models.items():
            # ==============================
            # 1) 参加可否判定
            # ==============================
            #   - enabled == False → 参加しない
            #   - APIキーが無ければ参加しない
            #   - 通常は supports_mode(mode) でモード判定
            #   - ★ Hermes(旧) だけはデバッグのため「モードに関係なく常に参加」
            # ==============================
            if not ai.enabled:
                results[name] = {
                    "status": "disabled",
                    "text": "",
                    "usage": None,
                    "meta": {"mode": mode_key},
                    "error": "disabled_by_config",
                }
                continue

            if not ai.has_api_key():
                results[name] = {
                    "status": "disabled",
                    "text": "",
                    "usage": None,
                    "meta": {"mode": mode_key},
                    "error": "no_api_key",
                }
                continue

            if name == "hermes":
                # ★ デバッグ用: judge_mode に関係なく常に参加
                participates = True
            else:
                participates = ai.supports_mode(mode_key)

            if not participates:
                results[name] = {
                    "status": "disabled",
                    "text": "",
                    "usage": None,
                    "meta": {"mode": mode_key},
                    "error": "disabled_by_mode",
                }
                continue

            # ==============================
            # 2) 実際の呼び出し
            # ==============================
            try:
                # ここでは messages だけを渡す。mode など余計な kwargs は渡さない。
                raw_text, usage = ai.call(messages=messages)

                # 正規化
                if not isinstance(raw_text, str):
                    raw_text = "" if raw_text is None else str(raw_text)

                # gpt51 専用の空レスガード
                is_empty_error, norm_text = self._normalize_gpt51_guard(
                    name=name,
                    text=raw_text,
                    usage=usage,
                )
                if is_empty_error:
                    results[name] = {
                        "status": "error",
                        "text": "",
                        "usage": usage,
                        "meta": {"mode": mode_key},
                        "error": "empty_response",
                    }
                    continue

                results[name] = {
                    "status": "ok",
                    "text": norm_text,
                    "usage": usage,
                    "meta": {"mode": mode_key},
                    "error": None,
                }

            except Exception as e:
                # 例外はここで吸収して "status=error" として記録
                results[name] = {
                    "status": "error",
                    "text": "",
                    "usage": None,
                    "meta": {"mode": mode_key},
                    "error": str(e),
                }

        return results
