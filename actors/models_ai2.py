# actors/models_ai2.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import traceback

from llm.llm_manager import LLMManager


CompletionType = Union[Dict[str, Any], Tuple[Any, ...], str]


class ModelsAI2:
    """
    複数 LLM から一斉に回答案を集めるためのクラス（デバッグ強化版）。

    ✅ 重要方針（ここが肝）
    - enabled_models の “正” は Streamlit AI Manager（st.session_state["ai_manager"]["enabled_models"]）
      → UI設定と必ず一致させる
    - UI設定が読めない/存在しない場合のみ LLMManager 側の available/enabled を参照
    - _meta に「何を投げたか」「どこ由来で選んだか」を必ず残す
    - persona(JSON) の llm_request_defaults をモデル呼び出しに反映できる
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
        # None の場合は collect() の度に最新の UI/available を見て追従する
        self._enabled_models_override: Optional[List[str]] = (
            list(enabled_models) if enabled_models is not None else None
        )

        # model_family などの参照用（collect() 内で毎回 refresh）
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
    # 最優先：Streamlit AI Manager から enabled_models を読む
    # ---------------------------------------
    def _resolve_enabled_models_from_streamlit(self) -> Optional[List[str]]:
        """
        Streamlit UI（AI Manager）の enabled_models を最優先で使う。
        無い/読めない場合は None を返す。

        想定:
          st.session_state["ai_manager"]["enabled_models"] = {"gpt52": True, "gpt51": False, ...}
        """
        try:
            import streamlit as st  # type: ignore
        except Exception:
            return None

        try:
            ai_state = st.session_state.get("ai_manager")
            if not isinstance(ai_state, dict):
                return None

            enabled = ai_state.get("enabled_models")
            if not isinstance(enabled, dict):
                return None

            picked = [str(k) for k, v in enabled.items() if bool(v)]
            # UIで全部OFFにされてるケースもあり得るので、その場合は空listを返す（Noneではない）
            return picked
        except Exception:
            return None

    # ---------------------------------------
    # 代替：LLMManager.get_available_models() から enabled を決める
    # ---------------------------------------
    def _resolve_enabled_models_from_available(
        self,
        available_models: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """
        available_models を元に enabled を決める。
        ただし UI がある場合はそちらが“真”なので、これはフォールバック専用。
        """
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
            return results

        # ✅ まず「呼べるモデル一覧」を取得（キー有無も含む）
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

        # ✅ persona defaults の model_family 参照のために props も取る（失敗しても落とさない）
        try:
            self.model_props = self.llm_manager.get_model_props() or {}
            if not isinstance(self.model_props, dict):
                self.model_props = {}
        except Exception:
            self.model_props = {}

        # ✅ target_models の決定（優先順位）
        # 1) override（固定）
        from_ui: Optional[List[str]] = None

        if self._enabled_models_override is not None:
            target_models = [str(x) for x in self._enabled_models_override if str(x).strip()]
            source = "override"
        else:
            # 2) Streamlit AI Manager（最優先の真）
            from_ui = self._resolve_enabled_models_from_streamlit()
            if from_ui is not None:
                target_models = list(from_ui)
                source = "streamlit_ai_manager"
            else:
                # 3) available の enabled
                target_models = self._resolve_enabled_models_from_available(available)
                source = "llm_manager_available"

        # ✅ 安全策：UI/override で指定されたモデルが available に無ければ除外
        # （存在しないモデルを呼ぶのは無駄なので、ここで落とす）
        available_keys = set(available.keys())
        filtered = [m for m in target_models if m in available_keys]
        dropped = [m for m in target_models if m not in available_keys]
        target_models = filtered

        # ✅ _meta を必ず残す（デバッグの要）
        results["_meta"] = {
            "status": "ok",
            "source": source,  # override / streamlit_ai_manager / llm_manager_available
            "available_models": list(available.keys()),
            "enabled_models": list(target_models),
            "dropped_not_available": dropped,
            "enabled_override": list(self._enabled_models_override) if self._enabled_models_override else None,
            "enabled_from_ui": list(from_ui) if from_ui is not None else None,
            "mode_current": mode_current,
            "reply_length_mode": reply_length_mode,
        }

        if not target_models:
            results["_system"] = {
                "status": "error",
                "text": "",
                "error": "enabled_models is empty (after resolving + filtering by available_models)",
                "traceback": None,
            }
            return results

        # ✅ 各モデルに投げる
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
