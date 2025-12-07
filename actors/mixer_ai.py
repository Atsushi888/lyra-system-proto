# actors/mixer_ai.py
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import streamlit as st

from actors.emotion_ai import EmotionAI, EmotionResult
from actors.scene_ai import SceneAI
from actors.emotion.emotion_state import EmotionState


class MixerAI:
    """
    EmotionAI / SceneAI / 手動デバッグ値（ドキドキ💓パワー）などを統合し、
    ModelsAI2 へ渡す emotion_override ペイロードを組み立てるクラス。

    優先度（上書きの強さ）は以下の通り:

        1. dokipower_control.py からの手動デバッグ値
           （session_state["mixer_debug_emotion"]）
        2. EmotionAI の直近推定結果（llm_meta["emotion"]）
        3. EmotionAI の長期状態（llm_meta["emotion_long_term"]）
        4. SceneAI が返すシーン固有の emotion / world_state など
    """

    def __init__(
        self,
        *,
        state: Optional[Mapping[str, Any]] = None,
        emotion_ai: Optional[EmotionAI] = None,
        scene_ai: Optional[SceneAI] = None,
    ) -> None:
        # Streamlit あり／なし両対応
        if state is not None:
            self.state = state  # type: ignore[assignment]
        else:
            self.state = st.session_state  # type: ignore[assignment]

        self.emotion_ai = emotion_ai
        self.scene_ai = scene_ai or SceneAI(state=self.state)

    # -----------------------------
    # 内部ヘルパ
    # -----------------------------
    def _get_debug_emotion(self) -> Optional[Dict[str, Any]]:
        """
        dokipower_control.py から渡される手動デバッグ用 EmotionResult。

        st.session_state["mixer_debug_emotion"] に辞書として入っている想定。
        relationship_level / masking_degree などが追加されてもそのまま拾う。
        """
        try:
            data = self.state.get("mixer_debug_emotion")
        except Exception:
            data = None

        if isinstance(data, dict):
            return data
        return None

    def _get_last_emotion(self) -> Optional[Dict[str, Any]]:
        """
        llm_meta["emotion"] に保存されている直近ターンの EmotionResult を取得。
        """
        try:
            llm_meta = self.state.get("llm_meta") or {}
            emo = llm_meta.get("emotion")
        except Exception:
            emo = None

        if isinstance(emo, dict):
            return emo
        return None

    def _get_long_term_emotion(self) -> Optional[Dict[str, Any]]:
        """
        llm_meta["emotion_long_term"] に保存されている長期感情状態。
        relationship_level などはここから供給される想定。
        """
        try:
            llm_meta = self.state.get("llm_meta") or {}
            lt = llm_meta.get("emotion_long_term")
        except Exception:
            lt = None

        if isinstance(lt, dict):
            return lt
        return None

    # -----------------------------
    # 公開 API
    # -----------------------------
    def build_emotion_override(self) -> Dict[str, Any]:
        """
        ModelsAI2.collect() に渡す emotion_override ペイロードを構築する。

        返り値の例:

        {
            "world_state": {...},
            "scene_emotion": {...},
            "emotion": { ... EmotionState as dict ... },
            "emotion_source": "debug_dokipower" | "auto"
        }
        """
        override: Dict[str, Any] = {}

        # 3) SceneAI 側 payload（world_state / scene_emotion など）
        try:
            scene_payload = self.scene_ai.build_emotion_override_payload()
            if isinstance(scene_payload, dict):
                for k, v in scene_payload.items():
                    override[k] = v
        except Exception as e:
            override["scene_error"] = str(e)

        # 2) EmotionAI の直近結果 / 長期結果
        last_emo = self._get_last_emotion()
        long_term_emo = self._get_long_term_emotion()

        # 1) ドキドキ💓デバッグ（最優先）
        debug_emo = self._get_debug_emotion()

        # EmotionState へ統合
        emotion_state = EmotionState.from_sources(
            base=last_emo,
            long_term=long_term_emo,
            manual=None,          # 将来、手動調整スライダーなどを別途追加する場合に使用
            debug=debug_emo,
            source_hint="auto",
        )

        override["emotion"] = emotion_state.to_dict()
        override["emotion_source"] = emotion_state.source

        # ---------- デバッグ用: Mixer が何を見てどう統合したかを llm_meta に記録 ----------
        try:
            llm_meta = self.state.get("llm_meta") or {}
            llm_meta["mixer_debug"] = {
                "has_debug_emo": bool(debug_emo),
                "has_last_emo": bool(last_emo),
                "has_long_term_emo": bool(long_term_emo),
                "emotion_source": emotion_state.source,
                "override_keys": list(override.keys()),
                "emotion_state": emotion_state.to_dict(),
            }
            self.state["llm_meta"] = llm_meta
        except Exception:
            # デバッグ用なので、失敗してもアプリ本体は止めない
            pass
        # -------------------------------------------------------------------

        return override
