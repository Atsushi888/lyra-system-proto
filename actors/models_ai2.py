# actors/models_ai2.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from llm.llm_manager import LLMManager


CompletionType = Union[Dict[str, Any], Tuple[Any, ...], str]


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
    # 内部ヘルパ：LLM からの戻り値を正規化
    # ---------------------------------------
    @staticmethod
    def _normalize_completion(completion: CompletionType) -> Dict[str, Any]:
        """
        LLMManager.chat(...) の戻り値形式の揺れを吸収して、
        text / usage / raw を取り出すためのヘルパ。

        想定パターン:
        - dict 形式: {"text": "...", "usage": {...}, ...}
        - tuple 形式: (text, usage_dict?) など
        - 素の str: "answer text"
        """
        text: str = ""
        usage: Any = None
        raw: Any = None

        # dict パターン
        if isinstance(completion, dict):
            raw = completion
            text = (
                completion.get("text")
                or completion.get("content")
                or completion.get("message")
                or ""
            )
            usage = completion.get("usage")

        # tuple / list パターン（例: (text, usage)）
        elif isinstance(completion, (tuple, list)):
            raw = {"raw_tuple": completion}

            if len(completion) >= 1:
                first = completion[0]
                text = str(first) if first is not None else ""

            if len(completion) >= 2:
                usage = completion[1]

        # それ以外（str など）はそのまま文字列化
        else:
            raw = {"raw": completion}
            text = "" if completion is None else str(completion)

        return {"text": text, "usage": usage, "raw": raw}

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
                # LLMManager.chat(...) への呼び出し
                # ※ LLMManager 側の引数名は "model" を想定
                completion: CompletionType = self.llm_manager.chat(
                    model=model_name,
                    messages=messages,
                    **props.get("defaults", {}),
                )

                norm = self._normalize_completion(completion)
                text = norm["text"]
                usage = norm["usage"]
                raw = norm["raw"]

                results[model_name] = {
                    "status": "ok",
                    "text": text,
                    "raw": raw,
                    "usage": usage,
                    "error": None,
                    "mode_current": mode_current,
                    "emotion_override": emotion_override,
                }

            except Exception as e:
                # ここで例外を握りつぶしておくことで、
                # 他モデルに影響させずにエラー内容だけ記録する。
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
