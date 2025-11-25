# actors/models_ai2.py (debug version)
from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime


class ModelsAI2:
    """
    各 LLM（gpt51 / grok / gemini / hermes）が
    AnswerTalker から渡された messages を受け取り、
    LLMManager を通して回答を収集する。

    デバッグ用途:
        - hermes / hermes_new は judge_mode に関係なく毎回実行
        - hermes 系は常に per_model_mode = "erotic" で回す
    """

    def __init__(self, llm_manager):
        self.llm_manager = llm_manager
        self.model_props = llm_manager.get_model_props()

    # --------------------------------------------------------------
    # Normalize各AIの返却値
    # --------------------------------------------------------------
    def _normalize(self, name: str, raw: Any) -> Dict[str, Any]:
        """
        LLMManager.call_model() の返却値を共通形式へ正規化する。
        """

        if not isinstance(raw, dict):
            return {
                "status": "error",
                "text": "",
                "usage": None,
                "meta": {"error": "invalid return type"},
            }

        text = raw.get("text") or raw.get("reply_text") or ""
        usage = raw.get("usage")
        meta = raw.get("meta", {})

        return {
            "status": "ok",
            "text": text,
            "usage": usage,
            "meta": meta,
        }

    # --------------------------------------------------------------
    # デバッグ強化版 collect()
    # --------------------------------------------------------------
    def collect(self, messages: List[Dict[str, str]], mode: str = "normal") -> Dict[str, Any]:
        """
        全モデルを走査し、結果を辞書にまとめる。

        デバッグ版特例:
            - hermes / hermes_new は judge_mode に無関係で常時オン
            - hermes 系は always mode="erotic" を渡して LLM を回す
        """

        results: Dict[str, Any] = {}
        mode_key = mode or "normal"

        for name, props in self.model_props.items():
            enabled_cfg = props.get("enabled", True)

            # --------------------------
            # モデル ON/OFF 判定
            # --------------------------
            if name == "gpt4o":
                enabled = False  # 完全停止
            elif name in ("hermes", "hermes_new"):
                enabled = enabled_cfg  # judge_mode 無視
            else:
                enabled = enabled_cfg  # 通常

            if not enabled:
                results[name] = {
                    "status": "disabled",
                    "text": "",
                    "usage": None,
                    "meta": {"mode": mode_key, "error": "disabled_by_config"},
                }
                continue

            # --------------------------
            # モード強制
            # --------------------------
            if name in ("hermes", "hermes_new"):
                per_model_mode = "erotic"
            else:
                per_model_mode = mode_key

            # --------------------------
            # 呼び出し
            # --------------------------
            try:
                raw = self.llm_manager.call_model(
                    model_name=name,
                    messages=messages,
                    mode=per_model_mode,
                )
                norm = self._normalize(name, raw)
                norm_meta = norm.get("meta") or {}
                norm_meta["mode"] = per_model_mode
                norm_meta["timestamp"] = datetime.utcnow().isoformat()
                norm["meta"] = norm_meta

                results[name] = norm
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "text": "",
                    "usage": None,
                    "meta": {"mode": per_model_mode, "error": str(e)},
                }

        return results
