# actors/models_ai2.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from actors.llm_ai import LLMAIRegistry, LLMAI
from actors.emotion_modes.emotion_style_prompt import EmotionOverride


class ModelsAI2:
    """
    新マルチLLM集約クラス（LLMAI ベース）。

    - AnswerTalker から messages / judge_mode / emotion_override を受け取る
    - LLMAIRegistry に登録された AI 達を一巡させて結果を集約する
    - 戻り値は llm_meta["models"] 用の dict
    """

    def __init__(
        self,
        registry: Optional[LLMAIRegistry] = None,
    ) -> None:
        # レジストリ（未指定なら標準構成を生成）
        self.registry: LLMAIRegistry = registry or LLMAIRegistry.create_default()

    # ------------------------------------------------------
    # メイン処理
    # ------------------------------------------------------
    def collect(
        self,
        messages: List[Dict[str, str]],
        mode_current: str = "normal",
        emotion_override: Optional[EmotionOverride] = None,
    ) -> Dict[str, Any]:
        """
        全登録 LLM を叩いて回答を集める。

        - 参加可否の判定は LLMAI.should_answer()
        - 文章量などの細かい制御は各 LLMAI 実装側に委譲
        """
        mode_key = (mode_current or "normal").lower()
        results: Dict[str, Any] = {}

        # ★ dict の items() で (name, ai) のペアを取り出す
        for name, ai in self.registry.all().items():
            # ==== 参加可否フィルタ ====
            if (not ai.enabled) or (not ai.should_answer(mode_key)):
                results[name] = {
                    "status": "disabled",
                    "text": "",
                    "usage": None,
                    "meta": {
                        "mode": mode_key,
                    },
                    "error": "disabled_by_config_or_mode",
                }
                continue

            # ==== 呼び出しパラメータ組み立て ====
            call_kwargs: Dict[str, Any] = {
                "mode": mode_key,  # LLM 側で参照するかもしれないので一応渡す
            }

            # emotion_override をそのまま渡す
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
