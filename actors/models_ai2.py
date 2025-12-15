from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import traceback

from llm.llm_manager import LLMManager


CompletionType = Union[Dict[str, Any], Tuple[Any, ...], str]


class ModelsAI2:
    """
    複数 LLM から一斉に回答案を集めるためのクラス（デバッグ強化版）。

    ★ 方針
    - 例外が出ても「models が空」になることを絶対に防ぐ
    - 各モデルごとに status / error / traceback を必ず残す
    - Persona(JSON)の llm_request_defaults をモデル呼び出しに反映できるようにする
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        *,
        enabled_models: Optional[List[str]] = None,
        persona: Any = None,
    ) -> None:
        self.llm_manager = llm_manager
        self.persona = persona

        self.model_props: Dict[str, Dict[str, Any]] = llm_manager.get_model_props()

        if enabled_models is not None:
            self.enabled_models = list(enabled_models)
        else:
            self.enabled_models = [
                name for name, props in self.model_props.items()
                if props.get("enabled", True)
            ]

    # ---------------------------------------
    # 内部ヘルパ：LLM からの戻り値を正規化
    # ---------------------------------------
    @staticmethod
    def _normalize_completion(completion: CompletionType) -> Dict[str, Any]:
        text: str = ""
        usage: Any = None
        raw: Any = None

        if isinstance(completion, dict):
            raw = completion
            text = (
                completion.get("text")
                or completion.get("content")
                or completion.get("message")
                or ""
            )
            usage = completion.get("usage")

        elif isinstance(completion, (tuple, list)):
            raw = {"raw_tuple": completion}
            if completion:
                text = str(completion[0] or "")
            if len(completion) >= 2:
                usage = completion[1]

        else:
            raw = {"raw": completion}
            text = "" if completion is None else str(completion)

        return {"text": text, "usage": usage, "raw": raw}

    # ---------------------------------------
    # 内部ヘルパ：Persona defaults を取り出す
    # ---------------------------------------
    def _get_persona_call_defaults(self, model_name: str) -> Dict[str, Any]:
        """
        persona JSON の llm_request_defaults を、対象モデル名で引いて返す。

        想定:
          persona.raw["llm_request_defaults"]["gpt52"] = {...}

        persona が無い/形式が違う/未定義なら空dict。
        """
        p = self.persona
        if p is None:
            return {}

        raw = getattr(p, "raw", None)
        if not isinstance(raw, dict):
            return {}

        defs = raw.get("llm_request_defaults")
        if not isinstance(defs, dict):
            return {}

        # "gpt52" などモデル名キー直指定
        d = defs.get(model_name)
        if isinstance(d, dict):
            return dict(d)

        # フォールバック：model_family で拾いたい場合に備える（将来用）
        # 例: props[model_name]["extra"]["model_family"]
        try:
            extra = (self.model_props.get(model_name) or {}).get("extra") or {}
            fam = extra.get("model_family")
            if isinstance(fam, str) and fam in defs and isinstance(defs.get(fam), dict):
                return dict(defs[fam])
        except Exception:
            pass

        return {}

    # ---------------------------------------
    # 内部ヘルパ：kwargs を安全に合成
    # ---------------------------------------
    @staticmethod
    def _merge_kwargs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(base)
        for k, v in (override or {}).items():
            # None は「指定なし」として無視（潰さない）
            if v is None:
                continue
            out[k] = v
        return out

    # ---------------------------------------
    # メイン
    # ---------------------------------------
    def collect(
        self,
        messages: List[Dict[str, str]],
        *,
        mode_current: str = "normal",
        emotion_override: Optional[Dict[str, Any]] = None,
        reply_length_mode: str = "auto",
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        if not self.enabled_models:
            # ★ ここが重要：enabled_models 空でも必ず痕跡を残す
            results["_system"] = {
                "status": "error",
                "text": "",
                "error": "enabled_models is empty",
                "traceback": None,
            }
            return results

        for model_name in self.enabled_models:
            # Persona由来の呼び出しデフォルト（モデル別）
            persona_defaults = self._get_persona_call_defaults(model_name)

            # このcollect呼び出し側（AnswerTalker）から渡ってきたメタは
            # LLMへは直接渡さない（ただし将来必要ならここで制御可能）
            # → いまは persona_defaults のみを LLM kwargs として渡す。
            call_kwargs: Dict[str, Any] = dict(persona_defaults)

            try:
                completion: CompletionType = self.llm_manager.chat(
                    model=model_name,
                    messages=messages,
                    **call_kwargs,
                )

                norm = self._normalize_completion(completion)

                results[model_name] = {
                    "status": "ok",
                    "text": norm["text"],
                    "raw": norm["raw"],
                    "usage": norm["usage"],
                    "error": None,
                    "traceback": None,
                    "mode_current": mode_current,
                    "emotion_override": emotion_override,
                    "reply_length_mode": reply_length_mode,
                    "call_kwargs": call_kwargs,  # ★デバッグ用：実際に渡したkwargs
                }

            except Exception as e:
                results[model_name] = {
                    "status": "error",
                    "text": "",
                    "raw": None,
                    "usage": None,
                    "error": str(e),
                    "traceback": traceback.format_exc(limit=6),
                    "mode_current": mode_current,
                    "emotion_override": emotion_override,
                    "reply_length_mode": reply_length_mode,
                    "call_kwargs": call_kwargs,  # ★失敗時も残す
                }

        return results
