# actors/models_ai2.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from actors.llm_ai import LLMAIRegistry, LLMAI
from actors.emotion_modes.emotion_style_prompt import EmotionStyle


class ModelsAI2:
    """
    新マルチLLM集約クラス（LLMAI ベース）。

    - AnswerTalker から messages / mode_current / emotion_override を受け取る
    - emotion_override がある場合、system メッセージとして感情状態を注入する
    - LLMAIRegistry に登録された AI達を一巡させて結果を集約する
    """

    def __init__(
        self,
        llm_manager: Optional[Any] = None,
        registry: Optional[LLMAIRegistry] = None,
    ) -> None:
        self.llm_manager = llm_manager
        self.registry: LLMAIRegistry = registry or LLMAIRegistry.create_default()

    # ------------------------------------------------------
    # emotion_override を system メッセージに変換して注入
    # ------------------------------------------------------
    def _inject_emotion_into_messages(
        self,
        base_messages: List[Dict[str, str]],
        emotion_override: Optional[Dict[str, Any]],
        mode_key: str,
    ) -> List[Dict[str, str]]:
        if not emotion_override:
            return base_messages

        # 値取得（安全のため float() で落とす）
        mode = str(emotion_override.get("mode", mode_key))
        aff = float(emotion_override.get("affection", 0.0))
        aro = float(emotion_override.get("arousal", 0.0))
        ten = float(emotion_override.get("tension", 0.0))
        ang = float(emotion_override.get("anger", 0.0))
        sad = float(emotion_override.get("sadness", 0.0))
        exc = float(emotion_override.get("excitement", 0.0))

        # system メッセージとして注入
        emo_text = (
            "【感情オーバーライド適用】\n"
            "現在のキャラクターの感情状態を反映して返答してください。\n"
            f"- モード: {mode}\n"
            f"- 好意 / 親しみ: {aff:.2f}\n"
            f"- 興奮度（性的 / 情動）: {aro:.2f}\n"
            f"- 緊張・不安: {ten:.2f}\n"
            f"- 怒り: {ang:.2f}\n"
            f"- 悲しみ: {sad:.2f}\n"
            f"- 期待・ワクワク: {exc:.2f}\n"
            "\n"
            "返答の内容・語気・比喩・話題選択などに、これらの感情値を反映してください。"
        )

        injected = [{"role": "system", "content": emo_text}] + list(base_messages)
        return injected

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

        # ★★★ ここが今回の一番重要ポイント ★★★
        # messages に感情オーバーライド system メッセージを注入
        messages_injected = self._inject_emotion_into_messages(
            base_messages=messages,
            emotion_override=emotion_override,
            mode_key=mode_key,
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
