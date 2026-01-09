# actors/models_ai2.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import traceback

from llm.llm_manager import LLMManager

CompletionType = Union[Dict[str, Any], Tuple[Any, ...], str]


class ModelsAI2:
    """
    複数 LLM から一斉に回答案を集めるためのクラス（デバッグ強化版）。

    ✅ 方針
    - AI Manager の enabled を確実に反映（enabled=True のみ呼ぶ）
    - APIキーが必要なモデルは has_key=True のみ呼ぶ
    - enabled_models_override が指定されても、無効/鍵なしは除外（安全）
    - persona(JSON) の llm_request_defaults をモデル呼び出しに反映
    - _meta に「今回投げた/除外した」一覧を必ず残す
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

        # enabled_models を「固定したい場合のみ」保持
        # None の場合は collect() の度に最新状態に追従
        self._enabled_models_override: Optional[List[str]] = (
            list(enabled_models) if enabled_models is not None else None
        )

        # model_family 参照などのために保持（collect内で更新）
        self.model_props: Dict[str, Dict[str, Any]] = {}

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
    # 内部ヘルパ：Persona defaults を取り出す（モデル別）
    # ---------------------------------------
    def _get_persona_call_defaults(self, model_name: str) -> Dict[str, Any]:
        """
        persona.raw["llm_request_defaults"]["gpt52"] = {...} を想定。
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

        # 1) モデル名キー直指定（gpt52 など）
        d = defs.get(model_name)
        if isinstance(d, dict):
            return dict(d)

        # 2) フォールバック：model_family（将来用）
        try:
            extra = (self.model_props.get(model_name) or {}).get("extra") or {}
            fam = extra.get("model_family")
            if isinstance(fam, str) and fam in defs and isinstance(defs.get(fam), dict):
                return dict(defs[fam])
        except Exception:
            pass

        return {}

    # ---------------------------------------
    # 内部ヘルパ：None 値は落とす（安全）
    # ---------------------------------------
    @staticmethod
    def _drop_none_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in (d or {}).items() if v is not None}

    # ---------------------------------------
    # enabled_models 解決（AI Manager 追従）
    # ---------------------------------------
    def _resolve_target_models(
        self,
        props: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        props は get_available_models() の戻り値を想定（enabled/has_key を含む）
        Returns:
          (target_models, filtered_detail)
        """
        all_models = list((props or {}).keys())

        # override がある場合はそれを候補の基準にする（ただし後段で enabled/has_key で除外）
        if self._enabled_models_override is not None:
            candidates = [m for m in self._enabled_models_override if m in props]
        else:
            candidates = all_models

        filtered_disabled: List[str] = []
        filtered_no_key: List[str] = []
        target: List[str] = []

        for name in candidates:
            p = props.get(name) or {}
            enabled = bool(p.get("enabled", True))
            has_key = bool(p.get("has_key", True))

            if not enabled:
                filtered_disabled.append(name)
                continue
            if not has_key:
                filtered_no_key.append(name)
                continue

            target.append(name)

        return target, {
            "filtered_disabled": filtered_disabled,
            "filtered_no_key": filtered_no_key,
        }

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

        if not messages:
            results["_system"] = {
                "status": "error",
                "text": "",
                "error": "messages is empty",
                "traceback": None,
            }
            return results

        # ✅ 毎回最新を取得（enabled/has_key も含めて判断するため）
        try:
            self.model_props = self.llm_manager.get_available_models()
        except Exception as e:
            results["_system"] = {
                "status": "error",
                "text": "",
                "error": f"llm_manager.get_available_models() failed: {e}",
                "traceback": traceback.format_exc(limit=8),
            }
            return results

        target_models, filtered = self._resolve_target_models(self.model_props)

        # ✅ まず _meta を必ず残す
        results["_meta"] = {
            "status": "ok",
            "mode_current": mode_current,
            "reply_length_mode": reply_length_mode,
            "all_models": list((self.model_props or {}).keys()),
            "enabled_models_override": list(self._enabled_models_override) if self._enabled_models_override is not None else None,
            "target_models": list(target_models),
            **filtered,
        }

        if not target_models:
            results["_system"] = {
                "status": "error",
                "text": "",
                "error": "no enabled+available models after filtering (enabled/has_key)",
                "traceback": None,
            }
            return results

        for model_name in target_models:
            persona_defaults = self._get_persona_call_defaults(model_name)
            call_kwargs: Dict[str, Any] = self._drop_none_kwargs(dict(persona_defaults))

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
                    "call_kwargs": call_kwargs,
                }

            except Exception as e:
                results[model_name] = {
                    "status": "error",
                    "text": "",
                    "raw": None,
                    "usage": None,
                    "error": str(e),
                    "traceback": traceback.format_exc(limit=8),
                    "mode_current": mode_current,
                    "emotion_override": emotion_override,
                    "reply_length_mode": reply_length_mode,
                    "call_kwargs": call_kwargs,
                }

        return results
