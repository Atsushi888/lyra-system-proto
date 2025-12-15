from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import traceback

from llm.llm_manager import LLMManager


CompletionType = Union[Dict[str, Any], Tuple[Any, ...], str]


class ModelsAI2:
    """
    複数 LLM から一斉に回答案を集めるためのクラス（デバッグ強化版）。

    ★ 新方針（v0.5）
    - 実行対象モデルは「AIManagerの設定」を最優先で決める
      - enabled_models(map) / priority(list) / select_mode("Auto"/"Manual")
    - Manual: 有効モデルのうち priority 最上位 1つだけ実行
    - Auto  : 有効モデルを priority 順で全て実行
    - 実行されなかったモデル（disabled / not_selected）は results に痕跡を残す
    - 例外が出ても「results が空」になることを絶対に防ぐ
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        *,
        enabled_models: Optional[List[str]] = None,  # 旧互換：固定リスト指定（使うならこれが最優先）
    ) -> None:
        self.llm_manager = llm_manager
        self.model_props: Dict[str, Dict[str, Any]] = llm_manager.get_model_props() or {}

        # 旧互換：enabled_models が明示されていたら、そのリストを“候補集合”として使う
        if enabled_models is not None:
            self._fixed_enabled_models = list(enabled_models)
        else:
            self._fixed_enabled_models = None

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
    # 内部ヘルパ：実行対象モデルの決定
    # ---------------------------------------
    def _decide_active_models(
        self,
        *,
        select_mode: str,
        priority: Optional[List[str]],
        enabled_map: Optional[Dict[str, bool]],
    ) -> Dict[str, Any]:
        """
        Returns:
          {
            "all_models": [...props keys...],
            "ordered_models": [...priority applied...],
            "enabled_models": [...enabled only...],
            "active_models": [...actually executed...],
            "disabled_models": [...not enabled...],
            "not_selected_models": [...enabled but excluded by Manual...],
            "select_mode": "auto" | "manual",
          }
        """
        props = self.model_props or {}
        all_models: List[str] = list(props.keys())

        # priority の適用（未知は後ろへ）
        pri = list(priority or [])
        ordered: List[str] = [m for m in pri if m in props]
        for m in all_models:
            if m not in ordered:
                ordered.append(m)

        # enabled 判定（AIManager enabled_map があれば最優先）
        enabled_models: List[str] = []
        disabled_models: List[str] = []

        for m in ordered:
            # 旧互換：固定リストがあるなら、それ以外は候補から落とす（disabled扱いではなく “not_in_fixed”）
            if self._fixed_enabled_models is not None and m not in self._fixed_enabled_models:
                continue

            if isinstance(enabled_map, dict) and m in enabled_map:
                is_on = bool(enabled_map.get(m))
            else:
                # props の enabled を既定に
                is_on = bool((props.get(m) or {}).get("enabled", True))

            if is_on:
                enabled_models.append(m)
            else:
                disabled_models.append(m)

        mode = (select_mode or "Auto").strip().lower()
        if mode not in ("auto", "manual"):
            mode = "auto"

        # Manual は先頭1つだけ、Auto は全部
        if mode == "manual":
            active_models = enabled_models[:1]
            not_selected_models = enabled_models[1:]
        else:
            active_models = enabled_models[:]
            not_selected_models = []

        return {
            "all_models": all_models,
            "ordered_models": ordered,
            "enabled_models": enabled_models,
            "active_models": active_models,
            "disabled_models": disabled_models,
            "not_selected_models": not_selected_models,
            "select_mode": mode,
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
        # ★新：AIManager（または上位層）から渡す
        select_mode: str = "Auto",                    # "Auto" / "Manual"
        priority: Optional[List[str]] = None,         # 優先順位
        enabled_map: Optional[Dict[str, bool]] = None # enabled_models マップ
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

        decision = self._decide_active_models(
            select_mode=select_mode,
            priority=priority,
            enabled_map=enabled_map,
        )

        active_models: List[str] = decision["active_models"]
        disabled_models: List[str] = decision["disabled_models"]
        not_selected_models: List[str] = decision["not_selected_models"]
        ordered_models: List[str] = decision["ordered_models"]
        enabled_models: List[str] = decision["enabled_models"]
        select_mode_norm: str = decision["select_mode"]

        # まず system セクションに “今回の選定” を残す（デバッグ最重要）
        results["_system"] = {
            "status": "ok",
            "text": "",
            "error": None,
            "traceback": None,
            "selection": {
                "select_mode": select_mode_norm,
                "priority": list(priority or []),
                "ordered_models": ordered_models,
                "enabled_models": enabled_models,
                "active_models": active_models,
                "disabled_models": disabled_models,
                "not_selected_models": not_selected_models,
            },
        }

        # active が空なら、ここで明示的にエラー痕跡
        if not active_models:
            results["_system"]["status"] = "error"
            results["_system"]["error"] = "no_active_models"
            # disabled / not_selected も痕跡として入れる（Viewで見たい）
            for m in disabled_models:
                results[m] = {
                    "status": "disabled",
                    "text": "",
                    "raw": None,
                    "usage": None,
                    "error": "disabled_by_enabled_map",
                    "traceback": None,
                    "mode_current": mode_current,
                    "emotion_override": emotion_override,
                    "reply_length_mode": reply_length_mode,
                }
            for m in not_selected_models:
                results[m] = {
                    "status": "skipped",
                    "text": "",
                    "raw": None,
                    "usage": None,
                    "error": "skipped_by_manual_mode",
                    "traceback": None,
                    "mode_current": mode_current,
                    "emotion_override": emotion_override,
                    "reply_length_mode": reply_length_mode,
                }
            return results

        # disabled / skipped も results に残す（監査用）
        for m in disabled_models:
            results[m] = {
                "status": "disabled",
                "text": "",
                "raw": None,
                "usage": None,
                "error": "disabled_by_enabled_map",
                "traceback": None,
                "mode_current": mode_current,
                "emotion_override": emotion_override,
                "reply_length_mode": reply_length_mode,
            }
        for m in not_selected_models:
            results[m] = {
                "status": "skipped",
                "text": "",
                "raw": None,
                "usage": None,
                "error": "skipped_by_manual_mode",
                "traceback": None,
                "mode_current": mode_current,
                "emotion_override": emotion_override,
                "reply_length_mode": reply_length_mode,
            }

        # 実行本体
        for model_name in active_models:
            try:
                completion: CompletionType = self.llm_manager.chat(
                    model=model_name,
                    messages=messages,
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
                }

        return results
