# actors/mixer_ai.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional
import os

import streamlit as st

from actors.emotion_ai import EmotionAI
from actors.scene_ai import SceneAI

LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"


@dataclass
class MixerAI:
    """
    world_state / scene_emotion / Emotion 系の情報を集約して、
    PersonaBase.build_emotion_based_system_prompt() に渡すための
    emotion_override を構築するクラス。

    ここでのゴールはただ 1 つ：

        「UI から与えた world_state（特に others_present / 二人きり）と
         感情・関係性の数値を、素直に 1 つの dict にまとめて渡す」

    余計な“推測”や“上書き”は極力やらない。
    """

    state: Mapping[str, Any]
    emotion_ai: EmotionAI
    scene_ai: SceneAI

    def __init__(
        self,
        *,
        state: Optional[Mapping[str, Any]] = None,
        emotion_ai: EmotionAI,
        scene_ai: SceneAI,
    ) -> None:
        # state は AnswerTalker と同じものを共有する
        if state is not None:
            self.state = state
        else:
            self.state = st.session_state

        self.emotion_ai = emotion_ai
        self.scene_ai = scene_ai

    # ==========================================================
    # 内部ヘルパ
    # ==========================================================
    def _log(self, *args: Any) -> None:
        """LYRA_DEBUG=1 のときだけデバッグログを出す。"""
        if not LYRA_DEBUG:
            return
        try:
            st.write("[DEBUG:MixerAI]", *args)
        except Exception:
            # Streamlit 以外の環境でも死なないようにしておく
            pass

    def _get_llm_meta(self) -> Dict[str, Any]:
        meta = self.state.get("llm_meta")
        if not isinstance(meta, dict):
            meta = {}
        # セッション側へも戻しておく（初期化）
        self.state["llm_meta"] = meta  # type: ignore[index]
        return meta

    def _get_world_and_scene(self) -> Dict[str, Any]:
        """
        world_state / scene_emotion を取得する。

        優先順位：
          1) llm_meta に積まれているもの（AnswerTalker.speak で積んだもの）
          2) SceneAI からの最新版
        ※ DokiPowerControl などの UI が world_state["others_present"] を
           直接いじっている前提なので、ここでは絶対に潰さない。
        """
        meta = self._get_llm_meta()

        ws = meta.get("world_state")
        if not isinstance(ws, dict) or not ws:
            ws = self.scene_ai.get_world_state()

        scene_emo = meta.get("scene_emotion")
        if not isinstance(scene_emo, dict):
            scene_emo = self.scene_ai.get_scene_emotion(ws)

        # 念のため llm_meta にも反映
        meta["world_state"] = ws
        meta["scene_emotion"] = scene_emo
        self.state["llm_meta"] = meta  # type: ignore[index]

        self._log("world_state =", ws)
        self._log("scene_emotion =", scene_emo)

        return {
            "world_state": ws,
            "scene_emotion": scene_emo,
        }

    def _get_emotion_payload(self) -> Dict[str, Any]:
        """
        EmotionResult / 長期感情 / 手動オーバーライドをマージして、
        emotion_override["emotion"] に入れる dict を作る。

        ここでは「数値的な感情・関係性」を扱い、
        world_state（場所・時間・others_present）は触らない。
        """
        meta = self._get_llm_meta()

        # EmotionResult.to_dict() が llm_meta["emotion"] に入っている想定
        emo_cur = meta.get("emotion") or {}
        if not isinstance(emo_cur, dict):
            emo_cur = {}

        # 長期感情（あれば）を緩めにマージ
        emo_long = meta.get("emotion_long_term") or {}
        if not isinstance(emo_long, dict):
            emo_long = {}

        # 手動オーバーライド（DokiPowerControl 等が使うならここに積む）
        manual = meta.get("emotion_override_manual") or {}
        if not isinstance(manual, dict):
            manual = {}

        emotion: Dict[str, Any] = {}

        # 1) 長期 → 2) 現在 → 3) 手動オーバーライド の順で上書き
        for src in (emo_long, emo_cur, manual):
            for k, v in src.items():
                emotion[k] = v

        self._log("emotion (merged) =", emotion)

        return emotion

    # ==========================================================
    # 公開 API
    # ==========================================================
    def build_emotion_override(self) -> Dict[str, Any]:
        """
        PersonaBase.build_emotion_based_system_prompt() に渡す
        emotion_override を構築して返す。

        戻り値の形：
        {
            "world_state": {...},     # SceneAI + UI（DokiPowerControl） 由来
            "scene_emotion": {...},  # SceneManager 由来の場所ボーナス
            "emotion": {...},        # 感情・関係性の数値（現在＋長期＋手動）
        }
        """
        ws_scene = self._get_world_and_scene()
        world_state = ws_scene["world_state"]
        scene_emotion = ws_scene["scene_emotion"]

        emotion = self._get_emotion_payload()

        # ★ ここでは world_state["others_present"] を絶対に触らない ★
        # DokiPowerControl で True/False がセットされているなら、
        # そのまま Persona 側の system_prompt ビルドに渡される。

        payload = {
            "world_state": world_state,
            "scene_emotion": scene_emotion,
            "emotion": emotion,
        }

        self._log("emotion_override payload =", payload)
        return payload
