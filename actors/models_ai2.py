# actors/models_ai2.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union, Mapping
import traceback

try:
    import streamlit as st  # type: ignore
    _HAS_ST = True
except Exception:
    st = None  # type: ignore
    _HAS_ST = False

from llm.llm_manager import LLMManager


CompletionType = Union[Dict[str, Any], Tuple[Any, ...], str]


class ModelsAI2:
    """
    複数 LLM から一斉に回答案を集めるクラス（AI Manager 追従 + デバッグ強化）。

    ✅ 重要仕様
    - AI Manager（st.session_state["ai_manager"]["enabled_models"]）を「正」として毎回同期する
    - enabled=False のモデルは target から除外（＝呼ばない）
    - has_key=False（もし付与されていれば）も target から除外（＝呼ばない）
    - results["_meta"] は最後に入れる（NarratorManager のフォールバック事故回避）
    - persona(JSON) の llm_request_defaults をモデル呼び出しに反映できる
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        *,
        enabled_models: Optional[List[str]] = None,
        persona: Any = None,
        state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.llm_manager = llm_manager
        self.persona = persona
        self.state: Mapping[str, Any] = state or (st.session_state if _HAS_ST else {})

        # enabled_models を「固定したい場合のみ」保持
        self._enabled_models_override: Optional[List[str]] = (
            list(enabled_models) if enabled_models is not None else None
        )

        # 補助情報（model_family 等）参照用。collect() のたび refresh する。
        self.model_props: Dict[str, Dict[str, Any]] = {}

    # ---------------------------------------
    # 内部ヘルパ：AI Manager の enabled_models を同期
    # ---------------------------------------
    def _sync_enabled_from_ai_manager(self) -> Dict[str, Any]:
        """
        st.session_state["ai_manager"]["enabled_models"] を LLMManager に反映する。
        失敗しても落とさない。デバッグ用に結果を返す。
        """
        info: Dict[str, Any] = {"status": "skip", "applied": False}

        ai_state = None
        try:
            if _HAS_ST:
                ai_state = st.session_state.get("ai_manager")
            else:
                ai_state = self.state.get("ai_manager") if isinstance(self.state, dict) else None
        except Exception:
            ai_state = None

        if not isinstance(ai_state, dict):
            return info

        enabled = ai_state.get("enabled_models")
        if not isinstance(enabled, dict):
            return info

        try:
            if hasattr(self.llm_manager, "set_enabled_models"):
                self.llm_manager.set_enabled_models(enabled)  # type: ignore[attr-defined]
                info = {"status": "ok", "applied": True, "enabled_models": dict(enabled)}
        except Exception as e:
            info = {"status": "error", "applied": False, "error": str(e)}

        return info

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
    # None 値は落とす（安全）
    # ---------------------------------------
    @staticmethod
    def _drop_none_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in (d or {}).items() if v is not None}

    # ---------------------------------------
    # enabled_models 解決（AI Manager 追従）
    # ---------------------------------------
    def _resolve_target_models(
        self,
        *,
        model_props: Dict[str, Dict[str, Any]],
        available: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """
        呼び出すモデル一覧を決める（＝本当に呼ぶ）。
        - enabled=True のみ
        - has_key=False が分かる場合は除外
        - override がある場合はその順序を尊重（ただし enabled/has_key でフィルタ）
        """
        # enabled/has_key を判定できる map
        enabled_map: Dict[str, bool] = {}
        for name, props in (model_props or {}).items():
            if isinstance(props, dict):
                enabled_map[name] = bool(props.get("enabled", True))
            else:
                enabled_map[name] = True

        has_key_map: Dict[str, bool] = {}
        for name, props in (available or {}).items():
            if isinstance(props, dict):
                has_key_map[name] = bool(props.get("has_key", True))
            else:
                has_key_map[name] = True

        def is_allowed(name: str) -> bool:
            if enabled_map.get(name, True) is False:
                return False
            if has_key_map.get(name, True) is False:
                return False
            return True

        if self._enabled_models_override is not None:
            # 手動固定（AI Manager 追従しない）
            return [m for m in list(self._enabled_models_override) if is_allowed(m)]

        # 通常：model_props の enabled=True を採用
        targets: List[str] = []
        for name in (model_props or {}).keys():
            if is_allowed(name):
                targets.append(name)

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

        # 1) AI Manager の enabled_models を毎回同期（ここが「正」）
        sync_info = self._sync_enabled_from_ai_manager()

        # 2) 毎回最新の props を取得
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

        try:
            available = self.llm_manager.get_available_models() or {}
        except Exception:
            available = {}

        target_models = self._resolve_target_models(model_props=self.model_props, available=available)

        if not target_models:
            results["_system"] = {
                "status": "error",
                "text": "",
                "error": "enabled_models is empty (after AI Manager sync/filter)",
                "traceback": None,
            }
            # _meta は最後に入れるが、ここでは入れても順序は関係ない（モデルが無い）
            results["_meta"] = {
                "status": "error",
                "mode_current": mode_current,
                "reply_length_mode": reply_length_mode,
                "ai_manager_sync": sync_info,
                "target_models": [],
                "all_models": list((self.model_props or {}).keys()),
            }
            return results

        # 3) モデルごとに収集
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

        # 4) _meta は「最後」に入れる（Round0 フォールバック事故対策）
        results["_meta"] = {
            "status": "ok",
            "mode_current": mode_current,
            "reply_length_mode": reply_length_mode,
            "ai_manager_sync": sync_info,
            "target_models": list(target_models),
            "all_models": list((self.model_props or {}).keys()),
        }

        return results
