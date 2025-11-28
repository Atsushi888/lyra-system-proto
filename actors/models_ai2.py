from __future__ import annotations

from typing import Any, Dict, List, Optional

from actors.llm_ai import LLMAIRegistry, LLMAI
from actors.emotion_modes.emotion_style_prompt import EmotionStyle


class ModelsAI2:
    """
    新マルチLLM集約クラス（LLMAI ベース）。

    - AnswerTalker から messages / mode_current / emotion_override を受け取る
    - emotion_override がある場合、EmotionStyle から生成した system メッセージを
      既存のペルソナ system の直後に挿入して LLM に渡す
    - LLMAIRegistry に登録された AI 達を一巡させて結果を集約する
    """

    def __init__(
        self,
        llm_manager: Optional[Any] = None,
        registry: Optional[LLMAIRegistry] = None,
    ) -> None:
        self.llm_manager = llm_manager
        self.registry: LLMAIRegistry = registry or LLMAIRegistry.create_default()

    # ------------------------------------------------------
    # EmotionStyle を system メッセージとして注入
    # ------------------------------------------------------
    def _inject_emotion_into_messages(
        self,
        base_messages: List[Dict[str, str]],
        emotion_style: Optional[EmotionStyle],
    ) -> List[Dict[str, str]]:
        """
        EmotionStyle から生成した system プロンプトを、既存の system メッセージ
        （ペルソナ定義）の直後に挿入する。

        base_messages:
            Persona.build_messages() などで組み立てられた元の messages。
        """
        if emotion_style is None:
            return base_messages

        emo_system_msg = {
            "role": "system",
            "content": emotion_style.build_system_prompt(),
        }

        new_messages: List[Dict[str, str]] = []
        inserted = False

        for msg in base_messages:
            new_messages.append(msg)
            # 最初の system メッセージの直後に挿入
            if (not inserted) and msg.get("role") == "system":
                new_messages.append(emo_system_msg)
                inserted = True

        # 万一 system が一つも無かった場合は、先頭に挿入
        if not inserted:
            new_messages.insert(0, emo_system_msg)

        return new_messages

    # ------------------------------------------------------
    # メイン処理
    # ------------------------------------------------------
    def collect(
        self,
        messages: List[Dict[str, str]],
        mode_current: str = "normal",
        emotion_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        mode_key = (mode_current or "normal").lower()
        results: Dict[str, Any] = {}

        # emotion_override → EmotionStyle（LLMAI 互換）
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
                emotion_style = None

        # ★ EmotionStyle を system prompt として統合
        messages_injected = self._inject_emotion_into_messages(
            base_messages=messages,
            emotion_style=emotion_style,
        )

        # ------------------------------------------------------
        # 各 LLM を叩く
        # ------------------------------------------------------
        for name, ai in self.registry.all().items():

            # モード不一致なら skip
            if not ai.enabled or not ai.should_answer(mode_key):
                results[name] = {
                    "status": "disabled",
                    "text": "",
                    "usage": None,
                    "meta": {"mode": mode_key},
                    "error": "disabled_by_config_or_mode",
                }
                continue

            # 呼び出し引数
            call_kwargs: Dict[str, Any] = {"mode": mode_key}
            if getattr(ai, "max_tokens", None) is not None:
                call_kwargs["max_tokens"] = int(ai.max_tokens)

            # 互換のため EmotionStyle もそのまま渡す（サブクラス側が使えばさらに強く効く）
            if emotion_style is not None:
                call_kwargs["emotion_style"] = emotion_style

            # 実際の LLM 呼び出し
            try:
                text, usage = ai.call(messages_injected, **call_kwargs)

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
