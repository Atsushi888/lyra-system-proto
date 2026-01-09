# actors/models_ai2.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import traceback

from llm.llm_manager import LLMManager

CompletionType = Union[Dict[str, Any], Tuple[Any, ...], str]


class ModelsAI2:
    """
    複数 LLM から一斉に回答案を集めるためのクラス（デバッグ強化版）。

    ✅ 設計方針（重要）
    - enabled_models の解決は get_available_models() を正とする
      （AI Manager の “有効/無効” を確実に反映するため）
    - get_model_props() は補助情報（model_family 等）のために取得するに留める
    - persona(JSON) の llm_request_defaults をモデル呼び出しに反映できる
    - 例外が出ても results が空にならない（_system / _meta を必ず残す）
    - _meta に available / enabled / override / mode を必ず残す（デバッグの要）
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
        # None の場合は collect() の度に AI Manager の状態に追従する
        self._enabled_models_override: Optional[List[str]] = (
            list(enabled_models) if enabled_models is not None else None
        )

        # 補助情報（model_family 等）用：collect() のたびに refresh
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
    # 内部ヘルパ：None 値は落とす（安全）
    # ---------------------------------------
    @staticmethod
    def _drop_none_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in (d or {}).items() if v is not None}

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
        direct = defs.get(model_name)
        if isinstance(direct, dict):
            return dict(direct)

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
    # enabled_models 解決（AI Manager 追従の “正”）
    # ---------------------------------------
    def _resolve_enabled_models_from_available(
        self,
        available_models: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """
        get_available_models() の結果を正として enabled を決める。

        - override が指定されている場合はそれを最優先（固定運用）
        - それ以外は available_models の各 props の enabled を見てフィルタ
          （props が無い/壊れている場合は安全側で含める）
        """
        if self._enabled_models_override is not None:
            # 手動固定（AI Manager 追従しないモード）
            return [str(x) for x in self._enabled_models_override if str(x).strip()]

        enabled: List[str] = []
        for name, props in (available_models or {}).items():
            try:
                if props is None:
                    enabled.append(name)
                elif isinstance(props, dict):
                    if props.get("enabled", True):
                        enabled.append(name)
                else:
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
            results["_meta"] = {
                "status": "error",
                "available_models": [],
                "enabled_models": [],
                "enabled_override": list(self._enabled_models_override) if self._enabled_models_override else None,
                "mode_current": mode_current,
                "reply_length_mode": reply_length_mode,
            }
            return results

        # 1) available（AI Manager の enabled/has_key を反映する想定の正）
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
            results["_meta"] = {
                "status": "error",
                "available_models": [],
                "enabled_models": [],
                "enabled_override": list(self._enabled_models_override) if self._enabled_models_override else None,
                "mode_current": mode_current,
                "reply_length_mode": reply_length_mode,
            }
            return results

        # 2) props（model_family 等の補助情報。persona defaults のフォールバックに使う）
        try:
            props = self.llm_manager.get_model_props() or {}
            if not isinstance(props, dict):
                props = {}
            self.model_props = props
        except Exception:
            # ここは補助なので落とさない（persona defaults の fallback が弱くなるだけ）
            self.model_props = {}

        target_models = self._resolve_enabled_models_from_available(available)

        # ✅ まず _meta を必ず残す（「available」と「enabled」を分けて見せる）
        results["_meta"] = {
            "status": "ok",
            "available_models": list(available.keys()),
            "enabled_models": list(target_models),
            "enabled_override": list(self._enabled_models_override) if self._enabled_models_override else None,
            "mode_current": mode_current,
            "reply_length_mode": reply_length_mode,
        }

        if not target_models:
            results["_system"] = {
                "status": "error",
                "text": "",
                "error": "enabled_models is empty (after resolving from available_models)",
                "traceback": None,
            }
            return results

        # 3) 各モデル呼び出し
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
