# actors/mixer_ai.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import streamlit as st  # EmotionControl の値を読むため
    _HAS_ST = True
except Exception:
    st = None  # type: ignore
    _HAS_ST = False

from actors.emotion_ai import EmotionAI, EmotionResult


# ============================================
# シーン固有の感情補正値
# ============================================

@dataclass
class SceneEmotion:
    """
    シーン固有の感情補正値。
    すべて「-1.0〜+1.0」程度に収まるスケールを想定。

    scene_id:
        シーン識別用ID（"council_room", "sunset_hill" など）。
        ログやデバッグ用で、数値計算には使わない。
    mode:
        推奨モード（"normal" / "erotic" など）。
        空文字や "normal" なら特に上書きしない前提でもOK。
    """

    scene_id: str

    mode: str = "normal"
    affection: float = 0.0
    arousal: float = 0.0
    tension: float = 0.0
    anger: float = 0.0
    sadness: float = 0.0
    excitement: float = 0.0


# ============================================
# MixerAI 本体
# ============================================

class MixerAI:
    """
    フローリアの最終感情ベクトルを決定するミキサー。

    入力:
      - EmotionAI.last_short_result（短期感情）
      - SceneEmotion（シーン固有の補正）
      - EmotionControl（UI）の手動オーバーライド

    出力:
      - ModelsAI2.collect() にそのまま渡せる emotion_override dict:

        {
          "mode": str,
          "affection": float,
          "arousal": float,
          "tension": float,
          "anger": float,
          "sadness": float,
          "excitement": float,
        }
    """

    def __init__(
        self,
        emotion_ai: EmotionAI,
        *,
        use_streamlit_ui: bool = True,
        weight_short: float = 1.0,
        weight_scene: float = 1.0,
        weight_ui: float = 1.0,
    ) -> None:
        self.emotion_ai = emotion_ai
        self.use_streamlit_ui = use_streamlit_ui

        # 各ソースの重み（必要に応じて外から調整する）
        self.weight_short = float(weight_short)
        self.weight_scene = float(weight_scene)
        self.weight_ui = float(weight_ui)

    # ----------------------------------------
    # 補助: UI override 読み込み
    # ----------------------------------------
    def _load_ui_override(self) -> Optional[Dict[str, Any]]:
        """
        EmotionControl が st.session_state に保存している
        手動感情値を読み取る。

        戻り値:
          - None  : 何も指定なし（auto モードなど）
          - dict  : {"mode": ..., "affection": ..., ...}
        """
        if (not self.use_streamlit_ui) or (not _HAS_ST):
            return None

        mode = st.session_state.get("emotion_override_mode", "auto")
        manual = st.session_state.get("emotion_override_manual")

        # manual_full のときだけ MixerAI が UI 値を利用する
        if mode != "manual_full":
            return None

        if not isinstance(manual, dict):
            return None

        try:
            return {
                "mode": manual.get("mode", "normal"),
                "affection": float(manual.get("affection", 0.0)),
                "arousal": float(manual.get("arousal", 0.0)),
                "tension": float(manual.get("tension", 0.0)),
                "anger": float(manual.get("anger", 0.0)),
                "sadness": float(manual.get("sadness", 0.0)),
                "excitement": float(manual.get("excitement", 0.0)),
            }
        except Exception:
            # 数値化に失敗したら無視
            return None

    # ----------------------------------------
    # 補助: EmotionAI の短期感情
    # ----------------------------------------
    def _get_short_emotion(self) -> Optional[EmotionResult]:
        """
        EmotionAI が最後に解析した短期感情を返す。
        （なければ None）
        """
        short: Optional[EmotionResult] = getattr(
            self.emotion_ai, "last_short_result", None
        )
        return short

    # ----------------------------------------
    # 補助: スカラー合成
    # ----------------------------------------
    @staticmethod
    def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, v))

    def _mix_scalar(
        self,
        base: float,
        scene: float,
        ui: float,
    ) -> float:
        """
        1軸ぶんの合成。
        今は単純な線形結合 + クリップ。
        必要に応じて非線形カーブに差し替えてもよい。
        """
        val = (
            base * self.weight_short
            + scene * self.weight_scene
            + ui * self.weight_ui
        )
        return self._clamp(val)

    # ----------------------------------------
    # 公開API: ModelsAI2向け emotion_override を構築
    # ----------------------------------------
    def build_for_models(
        self,
        *,
        scene_emotion: Optional[SceneEmotion] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        ModelsAI2.collect() に渡す emotion_override dict を構築する。

        scene_emotion:
          シーン固有の感情補正。未指定なら None。
        """

        short = self._get_short_emotion()
        ui = self._load_ui_override()

        if short is None and scene_emotion is None and ui is None:
            # 何も情報がなければ override なし
            return None

        # ---- mode の決定優先順位: UI > Scene > short > "normal" ----
        mode = "normal"
        if ui and ui.get("mode"):
            mode = str(ui["mode"])
        elif scene_emotion and scene_emotion.mode:
            mode = scene_emotion.mode
        elif short and getattr(short, "mode", None):
            mode = short.mode  # type: ignore[assignment]

        # ---- 各ソースから値を取得（存在しないものは 0 扱い） ----
        def val_from_short(attr: str) -> float:
            return float(getattr(short, attr)) if short is not None else 0.0

        def val_from_scene(attr: str) -> float:
            return float(getattr(scene_emotion, attr)) if scene_emotion is not None else 0.0

        def val_from_ui(attr: str) -> float:
            return float(ui.get(attr, 0.0)) if ui is not None else 0.0

        affection = self._mix_scalar(
            val_from_short("affection"),
            val_from_scene("affection"),
            val_from_ui("affection"),
        )
        arousal = self._mix_scalar(
            val_from_short("arousal"),
            val_from_scene("arousal"),
            val_from_ui("arousal"),
        )
        tension = self._mix_scalar(
            val_from_short("tension"),
            val_from_scene("tension"),
            val_from_ui("tension"),
        )
        anger = self._mix_scalar(
            val_from_short("anger"),
            val_from_scene("anger"),
            val_from_ui("anger"),
        )
        sadness = self._mix_scalar(
            val_from_short("sadness"),
            val_from_scene("sadness"),
            val_from_ui("sadness"),
        )
        excitement = self._mix_scalar(
            val_from_short("excitement"),
            val_from_scene("excitement"),
            val_from_ui("excitement"),
        )

        return {
            "mode": mode,
            "affection": affection,
            "arousal": arousal,
            "tension": tension,
            "anger": anger,
            "sadness": sadness,
            "excitement": excitement,
        }
