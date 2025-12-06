# actors/models_ai2.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from llm.llm_manager import LLMManager


class ModelsAI2:
    """
    複数 LLM から一斉に回答案を集めるためのクラス。

    - LLMManager に登録されているモデル定義（model_props）を参照
    - enable フラグが立っているモデルだけを順番に呼び出し
    - 結果を {model_name: {...}} の dict で返す
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        *,
        enabled_models: Optional[List[str]] = None,
    ) -> None:
        self.llm_manager = llm_manager
        self.model_props: Dict[str, Dict[str, Any]] = llm_manager.get_model_props()

        if enabled_models is not None:
            self.enabled_models = enabled_models
        else:
            # model_props 内で `"enabled": True` なモデルだけを対象にする
            models: List[str] = []
            for name, props in self.model_props.items():
                if props.get("enabled", True):
                    models.append(name)
            self.enabled_models = models

    # ---------------------------------------
    # メイン：全モデルから回答を集める
    # ---------------------------------------
    def collect(
        self,
        messages: List[Dict[str, str]],
        *,
        mode_current: str = "normal",
        emotion_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        messages:
            OpenAI / OpenRouter 互換の chat messages 配列。
            ※ 先頭の system などは AnswerTalker 側で既に差し替え済み。
        mode_current:
            JudgeAI3 などと揃えるための「モード名」。temperature などの
            プリセット切り替えに使いたい場合はここで参照可能。
        emotion_override:
            MixerAI から渡された感情オーバーライド情報（任意）。
            ここでは logging 用にそのまま結果に混ぜるだけ。
        """

        results: Dict[str, Any] = {}

        for model_name in self.enabled_models:
            props = self.model_props.get(model_name, {})

            try:
                # 新 LLMManager: chat_completion(model=..., messages=...)
                # 旧 LLMManager: chat(model=..., messages=...)
                defaults = props.get("defaults", {}) or {}

                if hasattr(self.llm_manager, "chat_completion"):
                    completion = self.llm_manager.chat_completion(
                        model=model_name,
                        messages=messages,
                        **defaults,
                    )
                else:
                    # 後方互換用フォールバック
                    completion = self.llm_manager.chat(
                        model=model_name,
                        messages=messages,
                        **defaults,
                    )

                # completion は dict を想定（llm_manager 側の仕様に準拠）
                text = completion.get("text") or completion.get("content") or ""
                usage = completion.get("usage")
                error = None

                results[model_name] = {
                    "status": "ok",
                    "text": text,
                    "raw": completion,
                    "usage": usage,
                    "error": error,
                    "mode_current": mode_current,
                    "emotion_override": emotion_override,
                }

            except Exception as e:
                results[model_name] = {
                    "status": "error",
                    "text": "",
                    "raw": None,
                    "usage": None,
                    "error": str(e),
                    "mode_current": mode_current,
                    "emotion_override": emotion_override,
                }

        return results
