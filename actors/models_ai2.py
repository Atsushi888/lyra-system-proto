# actors/models_ai2.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import traceback

from llm.llm_manager import LLMManager


CompletionType = Union[Dict[str, Any], Tuple[Any, ...], str]


class ModelsAI2:
    """
    複数 LLM から一斉に回答案を集めるクラス（デバッグ強化）。

    重要仕様:
    - AI Manager の enabled を必ず尊重して「呼ぶモデル」を決める
    - has_key（APIキー有無）が取れる場合も尊重
    - _meta に resolved 情報を必ず残す
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
        # None の場合は collect() の度に最新の available_models を見て追従する
        self._enabled_models_override: Optional[List[str]] = (
            [str(x) for x in enabled_models] if enabled_models is not None else None
        )

        # デバッグ用：直近のモデル情報スナップショット
        self._available_props: Dict[str, Dict[str, Any]] = {}

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

        # 2) フォールバック：model_family（extraから拾う）
        try:
            extra = (self._available_props.get(model_name) or {}).get("extra") or {}
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
    # 内部ヘルパ：今回「呼ぶモデル」を解決（enabled/has_key尊重）
    # ---------------------------------------
    def _resolve_target_models(self, available: Dict[str, Dict[str, Any]]) -> List[str]:
        # override 指定があるなら最優先（ただし存在するものだけ）
        if self._enabled_models_override is not None:
            exist = set((available or {}).keys())
            return [m for m in self._enabled_models_override if m in exist]

        targets: List[str] = []
        for name, props in (available or {}).items():
            try:
                if not isinstance(props, dict):
                    targets.append(str(name))
                    continue

                if props.get("enabled", True) is False:
                    continue

                # get_available_models() には has_key が入る想定
                if "has_key" in props and props.get("has_key", True) is False:
                    continue

                targets.append(str(name))
            except Exception:
                # 異常値でも安全側：呼ぶ（落とすと原因追いにくい）
                targets.append(str(name))

        return targets

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

        # 1) available_models を取得（has_key が欲しい）
        try:
            self._available_props = self.llm_manager.get_available_models() or {}
        except Exception as e:
            results["_system"] = {
                "status": "error",
                "text": "",
                "error": f"llm_manager.get_available_models() failed: {e}",
                "traceback": traceback.format_exc(limit=8),
            }
            return results

        # 2) 「呼ぶモデル」を enabled/has_key で確定
        target_models = self._resolve_target_models(self._available_props)

        # 3) _meta は必ず残す（デバッグ最優先）
        results["_meta"] = {
            "status": "ok",
            "mode_current": mode_current,
            "reply_length_mode": reply_length_mode,
            "enabled_models_override": list(self._enabled_models_override) if self._enabled_models_override is not None else None,
            "available_models": list((self._available_props or {}).keys()),
            "target_models": list(target_models),
        }

        if not target_models:
            results["_system"] = {
                "status": "error",
                "text": "",
                "error": "target_models is empty (all disabled or missing keys)",
                "traceback": None,
            }
            return results

        # 4) 収集
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
