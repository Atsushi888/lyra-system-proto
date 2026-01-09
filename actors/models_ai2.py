# actors/models_ai2.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import traceback

from llm.llm_manager import LLMManager

CompletionType = Union[Dict[str, Any], Tuple[Any, ...], str]


class ModelsAI2:
    """
    複数 LLM から一斉に回答案を集めるクラス（デバッグ強化版 / AI Manager 追従版）。

    重要方針:
    - “呼んでいいモデル”の正は llm_manager.get_available_models()
      （has_key を含めて判定できるため）
    - ただし available_models は「一覧」なので enabled=False は必ず除外する
    - enabled_models が override で渡された場合は、その集合だけを狙う
      さらに available/enabled/has_key と交差させる（ゾンビ起動を封じる）
    - _meta に available/selected/dropped を必ず残す
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

        # override 指定がある場合は「固定」
        self._enabled_models_override: Optional[List[str]] = (
            [str(x) for x in enabled_models] if enabled_models is not None else None
        )

        # Persona defaults の fallback（model_family参照など）用に保持
        # collect() 内で毎回 refresh する
        self.model_props: Dict[str, Dict[str, Any]] = {}

    # ---------------------------------------
    # LLM 戻り値の正規化
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
    # Persona defaults（モデル別）
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

        # 1) モデル名キー直指定
        d = defs.get(model_name)
        if isinstance(d, dict):
            return dict(d)

        # 2) フォールバック：model_family
        try:
            extra = (self.model_props.get(model_name) or {}).get("extra") or {}
            fam = extra.get("model_family")
            if isinstance(fam, str) and isinstance(defs.get(fam), dict):
                return dict(defs[fam])
        except Exception:
            pass

        return {}

    @staticmethod
    def _drop_none_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in (d or {}).items() if v is not None}

    # ---------------------------------------
    # “呼んでいいモデル”を確定する（核心）
    # ---------------------------------------
    def _resolve_target_models(
        self,
        available: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Returns:
          (target_models, meta_detail)
        """
        available_names: List[str] = [str(k) for k in (available or {}).keys()]

        enabled_names: List[str] = []
        dropped: Dict[str, str] = {}

        # available の中から enabled & has_key を満たすものだけ残す
        for name in available_names:
            p = available.get(name, {}) if isinstance(available, dict) else {}
            if not isinstance(p, dict):
                enabled_names.append(name)
                continue

            if not p.get("enabled", True):
                dropped[name] = "disabled"
                continue

            if "has_key" in p and not bool(p.get("has_key", True)):
                dropped[name] = "missing_api_key"
                continue

            enabled_names.append(name)

        # override があるならそれを最優先。ただし enabled_names と交差。
        if self._enabled_models_override is not None:
            override = [str(x) for x in self._enabled_models_override if str(x).strip()]
            target: List[str] = []
            for m in override:
                if m in enabled_names:
                    target.append(m)
                else:
                    # “指定したのに呼べない”理由を可能な範囲で残す
                    if m not in available_names:
                        dropped[m] = "not_registered_or_not_available"
                    else:
                        # available にはいるが enabled/has_key で落ちているケース
                        dropped.setdefault(m, "filtered_out")
            return target, {
                "available_models": available_names,
                "enabled_models_after_filter": enabled_names,
                "override_models": override,
                "dropped_models": dropped,
                "resolution_mode": "override_intersection",
            }

        # override なし：enabled_names が target
        return enabled_names, {
            "available_models": available_names,
            "enabled_models_after_filter": enabled_names,
            "override_models": None,
            "dropped_models": dropped,
            "resolution_mode": "enabled_from_available",
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

        # 1) available を毎回取得（enabled/has_key もここに載る）
        try:
            available = self.llm_manager.get_available_models() or {}
            if not isinstance(available, dict):
                available = {}
        except Exception as e:
            results["_system"] = {
                "status": "error",
                "text": "",
                "error": f"llm_manager.get_available_models() failed: {e}",
                "traceback": traceback.format_exc(limit=8),
            }
            return results

        # 2) model_props は Persona defaults の model_family 参照用（補助）
        try:
            props = self.llm_manager.get_model_props() or {}
            self.model_props = props if isinstance(props, dict) else {}
        except Exception:
            self.model_props = {}

        # 3) target_models を確定（ここが “他AIが走る” を止める本丸）
        target_models, meta_detail = self._resolve_target_models(available)

        # 4) _meta を必ず残す
        results["_meta"] = {
            "status": "ok",
            "mode_current": mode_current,
            "reply_length_mode": str(reply_length_mode or "auto"),
            "target_models": list(target_models),
            **meta_detail,
        }

        if not target_models:
            results["_system"] = {
                "status": "error",
                "text": "",
                "error": "No target models resolved (all filtered or override mismatch)",
                "traceback": None,
            }
            return results

        # 5) 呼び出し
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
                    "reply_length_mode": str(reply_length_mode or "auto"),
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
                    "reply_length_mode": str(reply_length_mode or "auto"),
                    "call_kwargs": call_kwargs,
                }

        return results
