# actors/models_ai2.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import traceback

from llm.llm_manager import LLMManager


CompletionType = Union[Dict[str, Any], Tuple[Any, ...], str]


class ModelsAI2:
    """
    複数 LLM から一斉に回答案を集めるためのクラス（デバッグ強化版）。

    ✅ 改良方針（最小侵襲）
    - enabled_models を __init__ 時に固定しない（AI Manager 変更に追従）
    - 例外が出ても results が空になるのを防ぐ
    - 各モデルごとに status / error / traceback / call_kwargs を必ず残す
    - persona(JSON) の llm_request_defaults をモデル呼び出しに反映できる
    - _meta に「今回投げたモデル一覧」を必ず残す（デバッグの要）
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
        # None の場合は collect() の度に最新の model_props を見て enabled を解決する
        self._enabled_models_override: Optional[List[str]] = (
            list(enabled_models) if enabled_models is not None else None
        )

        # persona defaults の fallback で model_family を見るために一応保持（毎回 refresh もする）
        self.model_props: Dict[str, Dict[str, Any]] = self.llm_manager.get_model_props()

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
    def _resolve_enabled_models(self, model_props: Dict[str, Dict[str, Any]]) -> List[str]:
        if self._enabled_models_override is not None:
            return list(self._enabled_models_override)

        enabled: List[str] = []
        for name, props in (model_props or {}).items():
            try:
                if props.get("enabled", True):
                    enabled.append(name)
            except Exception:
                enabled.append(name)
        return enabled

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

        # ✅ 毎回最新を取得（enabled追従のため）
        try:
            self.model_props = self.llm_manager.get_model_props()
        except Exception as e:
            results["_system"] = {
                "status": "error",
                "text": "",
                "error": f"llm_manager.get_model_props() failed: {e}",
                "traceback": traceback.format_exc(limit=8),
            }
            return results

        target_models = self._resolve_enabled_models(self.model_props)

        # ✅ まず _meta を必ず残す（「何を投げたか」可視化）
        results["_meta"] = {
            "status": "ok",
            "enabled_models": list(target_models),
            "mode_current": mode_current,
            "reply_length_mode": reply_length_mode,
        }

        if not target_models:
            results["_system"] = {
                "status": "error",
                "text": "",
                "error": "enabled_models is empty",
                "traceback": None,
            }
            return results

        for model_name in target_models:
            # Persona defaults
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
