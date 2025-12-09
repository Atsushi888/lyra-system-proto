# actors/mixer_ai.py
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import streamlit as st


class MixerAI:
    """
    EmotionAI / SceneAI / 手動デバッグ値（ドキドキ💓パワー）などを統合し、
    ModelsAI2 へ渡す emotion_override ペイロードを組み立てるクラス。

    優先度（上書きの強さ）は以下の通り:

        1. dokipower_control.py からの手動デバッグ値
           （session_state["mixer_debug_emotion"]）
        2. EmotionAI の直近推定結果（llm_meta["emotion"]）
        3. EmotionAI の長期状態（llm_meta["emotion_long_term"]）
        4. SceneAI が返すシーン固有の world_state / scene_emotion など

    以前は EmotionState クラスに依存していたが、
    現在はシンプルな dict マージのみで構成する。
    """

    def __init__(
        self,
        *,
        state: Optional[Mapping[str, Any]] = None,
        emotion_ai: Any = None,
        scene_ai: Any = None,
    ) -> None:
        # Streamlit あり／なし両対応
        if state is not None:
            self.state = state  # type: ignore[assignment]
        else:
            self.state = st.session_state  # type: ignore[assignment]

        # 型は Any で受ける（このクラス内部ではほぼ使わない）
        self.emotion_ai = emotion_ai
        self.scene_ai = scene_ai

    # -----------------------------
    # 内部ヘルパ
    # -----------------------------
    def _get_llm_meta(self) -> Dict[str, Any]:
        try:
            meta = self.state.get("llm_meta")
        except Exception:
            meta = None

        if not isinstance(meta, dict):
            meta = {}
        return meta

    def _get_debug_emotion(self) -> Optional[Dict[str, Any]]:
        """
        dokipower_control.py から渡される手動デバッグ用 EmotionResult 相当の dict。

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

    # -----------------------------
    # relationship / masking フォールバック
    # -----------------------------
    @staticmethod
    def _ensure_relationship_fields(emotion: Dict[str, Any]) -> None:
        """
        EmotionModel.sync_relationship_fields が通っていないケースでも、
        最低限 relationship_level / relationship_stage / relationship_label /
        masking_degree を補完できるようにするフォールバック。

        - relationship_level : 0〜100 の float
        - relationship_stage : 文字列ラベル
            ("acquaintance" / "friendly" / "close_friends" /
             "dating" / "soulmate")
        - relationship_label : UI 向けの日本語ラベル
        - masking_degree     : 0〜1 の float

        ※ 旧仕様（0〜4 の数値ステージ）との互換も維持する。
        """

        # --- affection → fallback 用 ---
        try:
            aff = float(
                emotion.get("affection_with_doki", emotion.get("affection", 0.0)) or 0.0
            )
        except Exception:
            aff = 0.0

        if aff < 0.0:
            aff = 0.0
        if aff > 1.0:
            aff = 1.0

        # --- relationship_level ---
        if "relationship_level" not in emotion:
            level = aff * 100.0
        else:
            try:
                level = float(emotion.get("relationship_level", 0.0) or 0.0)
            except Exception:
                level = aff * 100.0

        # clamp
        if level < 0.0:
            level = 0.0
        if level > 100.0:
            level = 100.0
        emotion["relationship_level"] = level

        # --- relationship_stage（内部ステージ名） ---
        raw_stage = emotion.get("relationship_stage", None)
        stage_name: str

        if isinstance(raw_stage, (int, float)):
            # 旧仕様: 0〜4 の数値インデックス
            idx = int(raw_stage)
            if idx <= 0:
                stage_name = "acquaintance"
            elif idx == 1:
                stage_name = "friendly"
            elif idx == 2:
                stage_name = "close_friends"
            elif idx == 3:
                stage_name = "dating"
            else:
                stage_name = "soulmate"
        elif isinstance(raw_stage, str) and raw_stage.strip():
            # 新仕様: すでに文字列ラベルが入っている場合はそのまま使う
            stage_name = raw_stage.strip()
        else:
            # 何もなければ relationship_level から推定
            if level >= 85.0:
                stage_name = "soulmate"
            elif level >= 60.0:
                stage_name = "dating"
            elif level >= 30.0:
                stage_name = "close_friends"
            elif level >= 10.0:
                stage_name = "friendly"
            else:
                stage_name = "acquaintance"

        emotion["relationship_stage"] = stage_name

        # --- relationship_label（UI 向け日本語ラベル） ---
        if not str(emotion.get("relationship_label", "")).strip():
            jp_label_map = {
                "acquaintance": "顔見知り〜クラスメイト",
                "friendly": "仲の良い先輩後輩",
                "close_friends": "親友〜両想い手前",
                "dating": "恋人同士",
                "soulmate": "将来を真剣に考える相手",
            }
            emotion["relationship_label"] = jp_label_map.get(
                stage_name, stage_name
            )

        # --- masking_degree（関係が深いほど「隠さない」＝小さく） ---
        if "masking_degree" in emotion:
            try:
                m = float(emotion.get("masking_degree", 0.0) or 0.0)
            except Exception:
                m = 0.0
            if m < 0.0:
                m = 0.0
            if m > 1.0:
                m = 1.0
            emotion["masking_degree"] = m
        else:
            # 単純なフォールバック: 深いほど素直になる
            masking = 1.0 - (level / 100.0)
            if masking < 0.0:
                masking = 0.0
            if masking > 1.0:
                masking = 1.0
            emotion["masking_degree"] = masking

    # -----------------------------
    # 公開 API
    # -----------------------------
    def build_emotion_override(self) -> Dict[str, Any]:
        """
        ModelsAI2.collect() に渡す emotion_override ペイロードを構築する。

        返り値の基本構造:

        {
            "world_state": {...},
            "scene_emotion": {...},
            "emotion": {...},            # EmotionResult.to_dict() 相当
            "emotion_long_term": {...}, # LongTermEmotion.to_dict() 相当
            "emotion_source": "debug" | "auto" | "none",
        }
        """
        llm_meta = self._get_llm_meta()

        world_state = llm_meta.get("world_state") or {}
        scene_emotion = llm_meta.get("scene_emotion") or {}
        short_emo = llm_meta.get("emotion") or {}
        long_term_emo = llm_meta.get("emotion_long_term") or {}

        # world_state / scene_emotion が空なら SceneAI から再取得してみる
        if (not world_state or not scene_emotion) and self.scene_ai is not None:
            try:
                payload = self.scene_ai.build_emotion_override_payload()
            except Exception:
                payload = {}
            if isinstance(payload, dict):
                world_state = world_state or payload.get("world_state") or {}
                scene_emotion = scene_emotion or payload.get("scene_emotion") or {}

        # デバッグ用 emotion（最優先）
        debug_emo = self._get_debug_emotion()

        if isinstance(debug_emo, dict) and debug_emo:
            emotion = dict(debug_emo)
            emotion_source = "debug"
        elif isinstance(short_emo, dict) and short_emo:
            emotion = dict(short_emo)
            emotion_source = "auto"
        else:
            emotion = {}
            emotion_source = "none"

        # relationship_level / masking_degree などの最低限補完
        if emotion:
            self._ensure_relationship_fields(emotion)

        override: Dict[str, Any] = {
            "world_state": world_state,
            "scene_emotion": scene_emotion,
            "emotion": emotion,
            "emotion_long_term": long_term_emo if isinstance(long_term_emo, dict) else {},
            "emotion_source": emotion_source,
        }

        # ---------- デバッグ用: Mixer が何を見てどう統合したかを llm_meta に記録 ----------
        try:
            llm_meta["mixer_debug"] = {
                "has_debug_emo": bool(debug_emo),
                "has_short_emo": isinstance(short_emo, dict) and bool(short_emo),
                "has_long_term_emo": isinstance(long_term_emo, dict) and bool(long_term_emo),
                "emotion_source": emotion_source,
                "override_keys": list(override.keys()),
                "emotion_preview": {
                    "affection": emotion.get("affection"),
                    "affection_with_doki": emotion.get("affection_with_doki"),
                    "relationship_level": emotion.get("relationship_level"),
                    "relationship_stage": emotion.get("relationship_stage"),
                    "relationship_label": emotion.get("relationship_label"),
                    "masking_degree": emotion.get("masking_degree"),
                } if emotion else {},
            }
            self.state["llm_meta"] = llm_meta
        except Exception:
            # デバッグ用なので、失敗してもアプリ本体は止めない
            pass
        # -------------------------------------------------------------------

        return override
