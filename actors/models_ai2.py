from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import traceback

from llm.llm_manager import LLMManager


CompletionType = Union[Dict[str, Any], Tuple[Any, ...], str]


class ModelsAI2:
    """
    複数 LLM から一斉に回答案を集めるためのクラス（デバッグ強化版 + Persona注入 + 動的パラメータ調整）。

    ★ 方針
    - 例外が出ても「models が空」になることを絶対に防ぐ
    - 各モデルごとに status / error / traceback を必ず残す
    - Persona(JSON)の llm_request_defaults をモデル別に読み取り、呼び出し kwargs に注入する
    - emotion_override / reply_length_mode / mode_current から「動的 override」を作って合成する
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
    # Persona(JSON) からモデル別 defaults を取得
    # ---------------------------------------
    def _get_persona_llm_defaults_for(self, model_name: str) -> Dict[str, Any]:
        """
        persona.raw["llm_request_defaults"][model_name] を返す。
        存在しない/壊れている場合は空 dict。

        include_reasoning 等の “Lyra用スイッチ” はここでは落とさず、sanitizeで処理する。
        """
        if self.persona is None:
            return {}

        raw = getattr(self.persona, "raw", None)
        if not isinstance(raw, dict):
            return {}

        lrd = raw.get("llm_request_defaults")
        if not isinstance(lrd, dict):
            return {}

        block = lrd.get(model_name)
        if not isinstance(block, dict):
            return {}

        return dict(block)

    # ---------------------------------------
    # 動的 override を算出（ここが本題）
    # ---------------------------------------
    @staticmethod
    def _safe_float(x: Any, default: float) -> float:
        try:
            return float(x)
        except Exception:
            return default

    @staticmethod
    def _safe_int(x: Any, default: int) -> int:
        try:
            return int(x)
        except Exception:
            return default

    def _build_dynamic_overrides(
        self,
        *,
        model_name: str,
        mode_current: str,
        emotion_override: Optional[Dict[str, Any]],
        reply_length_mode: str,
        base_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        いまの Lyra の状態から「呼び出しパラメータの微調整」を作る。

        前提：
        - Adapter/SDKが未対応のキーは OpenAIChatAdapter が TypeError 経由で落として再試行する想定
        - なのでここは “攻めてOK”、ただし過剰に壊すキーは避ける
        """
        overrides: Dict[str, Any] = {}

        # まず対象モデルだけ（今は gpt52 を主戦場に）
        if model_name not in ("gpt52", "gpt51", "gpt4o"):
            return overrides

        eo = emotion_override if isinstance(emotion_override, dict) else {}

        # Mixer が入れている想定のキー（無くても安全に）
        doki = self._safe_float(eo.get("doki_power"), 0.0)         # 0..1 想定
        masking = self._safe_float(eo.get("masking_level"), 30.0)  # 0..100 想定が多い
        env = str(eo.get("environment") or "")                     # "alone" / "with_others"
        interaction_hint = str(eo.get("interaction_mode_hint") or "auto")

        # normalize
        doki = max(0.0, min(1.0, doki))
        masking = max(0.0, min(100.0, masking))

        # ======================================================
        # 1) reply_length_mode → verbosity / tokens
        # ======================================================
        rlm = (reply_length_mode or "auto").lower()
        if rlm == "short":
            overrides["verbosity"] = "low"
            # 既に max 系があれば尊重しつつ、無いなら低め
            if "max_tokens" not in base_kwargs and "max_completion_tokens" not in base_kwargs:
                overrides["max_completion_tokens"] = 280
        elif rlm in ("normal",):
            overrides["verbosity"] = "medium"
            if "max_tokens" not in base_kwargs and "max_completion_tokens" not in base_kwargs:
                overrides["max_completion_tokens"] = 520
        elif rlm in ("long",):
            overrides["verbosity"] = "high"
            if "max_tokens" not in base_kwargs and "max_completion_tokens" not in base_kwargs:
                overrides["max_completion_tokens"] = 900
        elif rlm in ("story",):
            overrides["verbosity"] = "high"
            if "max_tokens" not in base_kwargs and "max_completion_tokens" not in base_kwargs:
                overrides["max_completion_tokens"] = 1400

        # ======================================================
        # 2) doki / masking / environment → temperature / penalties / reasoning
        # ======================================================
        # 基準（既存があればそれを土台に微調整）
        base_temp = self._safe_float(base_kwargs.get("temperature"), 0.7)
        base_presence = self._safe_float(base_kwargs.get("presence_penalty"), 0.0)
        base_freq = self._safe_float(base_kwargs.get("frequency_penalty"), 0.0)

        # doki が高いほど、表現を少し生き生きさせる（温度 + presence）
        # masking が高いほど、抑えて整える（温度 -）
        temp = base_temp + (doki * 0.20) - ((masking / 100.0) * 0.25)
        temp = max(0.2, min(1.1, temp))

        presence = base_presence + (doki * 0.25)
        presence = max(-0.5, min(1.2, presence))

        # 反復は少し抑える（会話がくどくなりやすいので）
        freq = base_freq + 0.05
        freq = max(-0.5, min(1.0, freq))

        # 人前 or with_others は少し落ち着かせる
        if env == "with_others" or interaction_hint in ("pair_public", "solo_with_others", "auto_with_others"):
            temp = max(0.2, temp - 0.08)

        overrides["temperature"] = round(temp, 3)
        overrides["presence_penalty"] = round(presence, 3)
        overrides["frequency_penalty"] = round(freq, 3)

        # reasoning は “必要なときだけ” 上げる（常時だとテンポが重くなる）
        # - normal系は low
        # - story/long で少し上げる
        # - ただし include_reasoning=false が Persona で明示されてたら sanitizeで落とされる
        if rlm in ("story", "long"):
            overrides["reasoning"] = "medium"
        else:
            overrides["reasoning"] = "low"

        # ======================================================
        # 3) mode_current の特例（将来拡張しやすい）
        # ======================================================
        mc = (mode_current or "").lower()
        if "narrator" in mc:
            # ナレーター系は安定寄り（暴れない）
            overrides["temperature"] = min(overrides.get("temperature", temp), 0.65)
            overrides["verbosity"] = overrides.get("verbosity", "medium")
            overrides["reasoning"] = "low"

        if "erotic" in mc:
            # 露骨にしない前提で、空気感の描写を少し増やす
            overrides["verbosity"] = "high" if rlm != "short" else "medium"
            overrides["temperature"] = min(max(overrides.get("temperature", temp), 0.55), 0.95)

        return overrides

    # ---------------------------------------
    # sanitize（外へ出す直前の最終整形）
    # ---------------------------------------
    @staticmethod
    def _sanitize_call_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapter/SDKに渡す直前の最終整形。

        - Lyra用の内部キーを落とす
        - include_reasoning=false なら reasoning を落とす
        """
        out = dict(kwargs)

        # Lyra内部キー（絶対に外へ出さない）
        out.pop("mode_current", None)
        out.pop("emotion_override", None)
        out.pop("reply_length_mode", None)

        include_reasoning = out.pop("include_reasoning", None)
        if include_reasoning is False:
            out.pop("reasoning", None)

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
            results["_system"] = {
                "status": "error",
                "text": "",
                "error": "enabled_models is empty",
                "traceback": None,
            }
            return results

        for model_name in self.enabled_models:
            # 1) model props defaults
            props = self.model_props.get(model_name, {}) or {}
            props_params = props.get("params")
            base_kwargs: Dict[str, Any] = dict(props_params) if isinstance(props_params, dict) else {}

            # 2) Persona defaults
            persona_kwargs = self._get_persona_llm_defaults_for(model_name)

            # 3) merge: props -> persona
            merged = dict(base_kwargs)
            merged.update(persona_kwargs)

            # 4) dynamic overrides（最終優先）
            dyn = self._build_dynamic_overrides(
                model_name=model_name,
                mode_current=mode_current,
                emotion_override=emotion_override,
                reply_length_mode=reply_length_mode,
                base_kwargs=merged,
            )
            merged.update(dyn)

            # 5) 実際に投げるのは sanitize 後
            safe_kwargs = self._sanitize_call_kwargs(merged)

            try:
                completion: CompletionType = self.llm_manager.chat(
                    model=model_name,
                    messages=messages,
                    **safe_kwargs,
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
                    "call_kwargs": safe_kwargs,       # 実際に投げた（sanitize後）
                    "props_kwargs": base_kwargs,      # props.params
                    "persona_kwargs": persona_kwargs, # Persona由来
                    "dynamic_kwargs": dyn,            # 動的 override（これが見たい）
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
                    "call_kwargs": safe_kwargs,
                    "props_kwargs": base_kwargs,
                    "persona_kwargs": persona_kwargs,
                    "dynamic_kwargs": dyn,
                }

        return results
