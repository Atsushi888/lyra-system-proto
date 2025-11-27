# actors/models_ai2.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from actors.llm_ai import LLMAIRegistry
from actors.emotion_modes.emotion_style_prompt import EmotionStyle


class ModelsAI2:
    """
    新マルチLLM集約クラス（LLMAI ベース）。

    - AnswerTalker から messages / judge_mode / emotion_style を受け取る
    - LLMAIRegistry に登録された AI 達を一巡させて結果を集約する
    - 戻り値は llm_meta["models"] 相当の dict
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
        emotion_style: Optional[EmotionStyle] = None,
    ) -> Dict[str, Any]:
        """
        全登録 LLM を叩いて回答を集める。

        - 参加可否の判定は LLMAI.should_answer()
        - emotion_style が指定されていれば、そのまま各 AI に渡す
        """
        mode_key = (mode_current or "normal").lower()
        results: Dict[str, Any] = {}

        # LLMAIRegistry.all() は List[LLMAI] を返す想定
        for ai in self.registry.all():
            name = ai.name

            # ==== 参加可否フィルタ ====
            if not ai.should_answer(mode_key):
                results[name] = {
                    "status": "disabled",
                    "text": "",
                    "usage": None,
                    "meta": {"mode": mode_key},
                    "error": "disabled_by_config_or_mode",
                }
                continue

            # ==== 共通 kwargs を構築 ====
            call_kwargs: Dict[str, Any] = {
                "mode": mode_key,
            }

            # 感情スタイルをそのまま渡す（対応 AI だけが使う）
            if emotion_style is not None:
                call_kwargs["emotion_style"] = emotion_style

            # ==== 実コール ====
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
