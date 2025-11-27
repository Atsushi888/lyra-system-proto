# actors/models_ai2.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from actors.llm_ai import LLMAIRegistry, LLMAI
from actors.emotion_modes.emotion_style_prompt import EmotionStyle


class ModelsAI2:
    """
    新マルチLLM集約クラス（LLMAI ベース）。

    - AnswerTalker から messages / mode_current / emotion_override を受け取る
    - LLMAIRegistry に登録された AI 達を一巡させて結果を集約する
    - 戻り値は llm_meta["models"] 用の dict
    """

    def __init__(
        self,
        llm_manager: Optional[Any] = None,
        registry: Optional[LLMAIRegistry] = None,
    ) -> None:
        # いまのところ llm_manager は未使用だが、将来 EmotionAI 連携で使う余地を残す
        self.llm_manager = llm_manager
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
        - 文章量は llm_ai.py 側で設定した ai.max_tokens をそのまま渡す
        - emotion_override があれば EmotionStyle に変換して各 AI に渡す
        """
        mode_key = (mode_current or "normal").lower()
        results: Dict[str, Any] = {}

        # emotion_override -> EmotionStyle (今は全モデル共通で使う)
        emotion_style: Optional[EmotionStyle] = None
        if emotion_override:
            try:
                emotion_style = EmotionStyle(
                    mode=str(emotion_override.get("mode", mode_key)),
                    affection=float(emotion_override.get("affection", 0.0)),
                    arousal=float(emotion_override.get("arousal", 0.0)),
                    tension=float(emotion_override.get("tension", 0.0)),
                    sadness=float(emotion_override.get("sadness", 0.0)),
                    excitement=float(emotion_override.get("excitement", 0.0)),
                )
            except Exception:
                # もし変換に失敗したら、素直に無視して通常モードで進める
                emotion_style = None

        for name, ai in self.registry.all().items():
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

            # === 呼び出しパラメータを組み立て ===
            call_kwargs: Dict[str, Any] = {"mode": mode_key}
            if getattr(ai, "max_tokens", None) is not None:
                call_kwargs["max_tokens"] = int(ai.max_tokens)

            # 感情スタイルを渡せる AI には渡す想定（未対応なら無視されるだけ）
            if emotion_style is not None:
                call_kwargs["emotion_style"] = emotion_style

            try:
                text, usage = ai.call(messages, **call_kwargs)

                results[name] = {
                    "status": "ok",
                    "text": text or "",
                    "usage": usage or {},
                    "meta": {
                        "mode": mode_key,
                        "emotion_override": bool(emotion_override),
                    },
                }
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "text": "",
                    "usage": {},
                    "meta": {
                        "mode": mode_key,
                        "emotion_override": bool(emotion_override),
                    },
                    "error": str(e),
                }

        return results
