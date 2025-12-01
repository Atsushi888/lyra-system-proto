# actors/mixer_ai.py
from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional

from actors.emotion_ai import EmotionAI
from actors.scene_ai import SceneAI


class MixerAI:
    """
    感情値を「混ぜる」役割をまとめたクラス。

    入力ソース:
      - EmotionAI.last_short_result  : 前ターンの短期感情
      - emotion_override_manual      : 手動パネルの値（state 経由）
      - SceneAI.get_scene_info()     : シーンごとの感情ボーナス

    出力:
      - ModelsAI2.collect() に渡す emotion_override(dict) を返す。
        {
          "mode": "normal" / "erotic" / ...,
          "affection": float,
          "arousal": float,
          ...
        }
    """

    EMOTION_KEYS = [
        "affection",
        "arousal",
        "tension",
        "anger",
        "sadness",
        "excitement",
    ]

    def __init__(
        self,
        *,
        emotion_ai: EmotionAI,
        scene_ai: SceneAI,
        state: Optional[MutableMapping[str, Any]] = None,
    ) -> None:
        self.emotion_ai = emotion_ai
        self.scene_ai = scene_ai
        self.state: MutableMapping[str, Any] = state or {}

    # ---------------------------------------------------------
    # 内部ヘルパ
    # ---------------------------------------------------------
    def _get_state(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    @staticmethod
    def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    # ---------------------------------------------------------
    # 公開 API
    # ---------------------------------------------------------
    def get_scene_emotion_bonus(self) -> Dict[str, float]:
        """
        SceneAI から emotion_bonus を取得して正規化する。
        """
        info = self.scene_ai.get_scene_info() or {}
        raw = info.get("emotion_bonus") or {}
        bonus: Dict[str, float] = {}
        for k in self.EMOTION_KEYS:
            try:
                v = float(raw.get(k, 0.0))
            except Exception:
                v = 0.0
            bonus[k] = self._clamp(v)
        return bonus

    def build_emotion_override_for_models(self) -> Optional[Dict[str, Any]]:
        """
        ModelsAI2.collect() に渡す emotion_override を構築する。

        emotion_override_mode:
          - "auto"        : EmotionAI の短期感情をベースに、Scene ボーナスを加算
          - "manual_full" : 手動パネルの値をベースに、Scene ボーナスを加算
        """
        mode_setting = str(self._get_state("emotion_override_mode", "auto"))
        manual = self._get_state("emotion_override_manual")
        scene_bonus = self.get_scene_emotion_bonus()

        base_mode = "normal"
        base_vals: Dict[str, float] = {k: 0.0 for k in self.EMOTION_KEYS}

        # ---------------------------
        # manual_full: 手動＋シーン
        # ---------------------------
        if mode_setting == "manual_full" and isinstance(manual, dict):
            base_mode = str(manual.get("mode", "normal"))
            for k in self.EMOTION_KEYS:
                try:
                    base_vals[k] = float(manual.get(k, 0.0))
                except Exception:
                    base_vals[k] = 0.0

        else:
            # ---------------------------
            # auto: EmotionAI の短期感情＋シーン
            # ---------------------------
            short = getattr(self.emotion_ai, "last_short_result", None)
            if short is not None:
                base_mode = getattr(short, "mode", "normal") or "normal"
                for k in self.EMOTION_KEYS:
                    base_vals[k] = float(getattr(short, k, 0.0))

        # Scene ボーナスを加算
        mixed: Dict[str, float] = {}
        for k in self.EMOTION_KEYS:
            v = base_vals.get(k, 0.0) + scene_bonus.get(k, 0.0)
            mixed[k] = self._clamp(v)

        # ベースも Scene も全部 0 なら、override 自体を省略してもよい
        if all(abs(v) < 1e-6 for v in mixed.values()) and base_mode == "normal":
            return None

        result: Dict[str, Any] = {"mode": base_mode}
        result.update(mixed)
        return result
