# actors/models_ai2.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from actors.llm_ai import LLMAIRegistry, LLMAI


class ModelsAI2:
    """
    新マルチLLM集約クラス（LLMAI ベース）。

    - AnswerTalker から messages / judge_mode / emotion_override を受け取る
    - LLMAIRegistry に登録された AI 達を一巡させて結果を集約する
    - 戻り値は llm_meta["models"] 用の dict
    """

    def __init__(
        self,
        llm_manager: Optional[Any] = None,
        registry: Optional[LLMAIRegistry] = None,
    ) -> None:
        # いまのところ llm_manager は未使用だが、将来拡張用に残しておく
        self.llm_manager = llm_manager
        # レジストリ（未指定なら標準構成を生成）
        self.registry: LLMAIRegistry = registry or LLMAIRegistry.create_default()

    # ------------------------------------------------------
    # メイン処理
    # ------------------------------------------------------
    def collect(
        self,
        messages: List[Dict[str, str]],
        mode_current: str = "normal",
        emotion_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        全登録LLMを叩いて回答を集める。

        - 参加可否の判定は LLMAI.should_answer()
        - 文章量ヒントは llm_ai.py 側で設定した ai.max_tokens をそのまま渡す
        - emotion_override があればそのまま ai.call(..., emotion_override=...) に渡す
          （サブクラス側で pop して使う／無視する）
        """
        mode_key = (mode_current or "normal").lower()
        results: Dict[str, Any] = {}

        for ai in self.registry.all():
            name = ai.name

            # ==== 参加可否フィルタ ====
            if (not ai.enabled) or (not ai.should_answer(mode_key)):
                results[name] = {
                    "status": "disabled",
                    "text": "",
                    "usage": None,
                    "meta": {"mode": mode_key},
                    "error": "disabled_by_config_or_mode",
                }
                continue

            call_kwargs: Dict[str, Any] = {
                "mode": mode_key,
            }

            # max_tokens ヒント
            if getattr(ai, "max_tokens", None) is not None:
                try:
                    call_kwargs["max_tokens"] = int(ai.max_tokens)
                except Exception:
                    pass

            # 感情オーバーライド（EmotionControl / EmotionAI）
            if emotion_override is not None:
                call_kwargs["emotion_override"] = emotion_override

            try:
                text, usage = ai.call(messages, **call_kwargs)
                results[name] = {
                    "status": "ok",
                    "text": text or "",
                    "usage": usage or {},
                    "meta": {
                        "mode": mode_key,
                    },
                    "error": None,
                }
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "text": "",
                    "usage": {},
                    "meta": {
                        "mode": mode_key,
                    },
                    "error": str(e),
                }

        return results
