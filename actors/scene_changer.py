# components/scene_changer.py
from __future__ import annotations

from typing import Dict, Any

import streamlit as st


class SceneChanger:
    """
    シーン移動と「シーンごとの感情ボーナス/ペナルティ」をまとめて扱う UI コンポーネント。

    - 現在シーンの表示
    - 移動先シーンの選択
    - シーンごとの Emotion ボーナス編集（affection / arousal / tension / anger / sadness / excitement）
    - 決定ボタンで current_scene を更新し、scene_emotion_mods に保存
    """

    # シーン一覧（とりあえず固定定義）
    SCENES: Dict[str, Dict[str, str]] = {
        "END": {
            "label": "会話を終了する",
            "description": "この選択肢を選ぶと、会話シーンを終了し、物語を一旦締めくくります。",
        },
        "TOWN": {
            "label": "街",
            "description": "人の賑わいと灯りに満ちた街。安全で、のんびり会話するのに向いています。",
        },
        "ROAD": {
            "label": "街道筋",
            "description": "街と街をつなぐ街道。旅の途中の少し開けた場所で、静かな時間が流れます。",
        },
        "CAVE": {
            "label": "封印の氷窟",
            "description": "冷気と静寂に包まれた洞窟。緊張感や神秘性の強いシーンに向いています。",
        },
    }

    # EmotionAI / MixerAI と同じ軸でボーナスを持つ想定
    EMOTION_KEYS = [
        "affection",
        "arousal",
        "tension",
        "anger",
        "sadness",
        "excitement",
    ]

    def __init__(self, *, session_key: str = "scene_state") -> None:
        self.session_key = session_key

        # scene_state の初期化
        if self.session_key not in st.session_state:
            # 初期シーンは「街」にしておく（好みで変えてOK）
            st.session_state[self.session_key] = {
                "current_scene": "TOWN",
                "pending_scene": "TOWN",
            }

        # シーンごとの感情ボーナス初期化
        if "scene_emotion_mods" not in st.session_state:
            st.session_state["scene_emotion_mods"] = self._create_default_mods()

    # ---------------------------------------------------------
    # 内部ヘルパ
    # ---------------------------------------------------------
    def _create_default_mods(self) -> Dict[str, Dict[str, float]]:
        """
        全シーン共通で 0.0 のボーナスを持つ dict を作る。
        """
        base = {k: 0.0 for k in self.EMOTION_KEYS}
        return {scene_id: dict(base) for scene_id in self.SCENES.keys()}

    @staticmethod
    def _get_emotion_label(key: str) -> str:
        labels = {
            "affection": "好意（affection）",
            "arousal": "昂り・色気（arousal）",
            "tension": "緊張感（tension）",
            "anger": "怒り（anger）",
            "sadness": "哀しさ（sadness）",
            "excitement": "わくわく感（excitement）",
        }
        return labels.get(key, key)

    @staticmethod
    def _get_emotion_help(key: str) -> str:
        helps = {
            "affection": "この場所にいるとき、キャラクターの『好意』をどれだけ増減させるか。",
            "arousal": "この場所が与える色気・高ぶりの強さ。プラスで色っぽく、マイナスで落ち着いた雰囲気に。",
            "tension": "緊張感。危険・シリアスな場所ではプラス、のどかな場所ではマイナス寄り。",
            "anger": "苛立ちや怒りが出やすい場所ならプラス、穏やかな場所ならマイナスに。",
            "sadness": "物悲しさ・寂しさを誘う場所ならプラス。",
            "excitement": "わくわく・高揚感を誘う場所ならプラス、落ち着きを与えるならマイナス寄り。",
        }
        return helps.get(key, "")

    # ---------------------------------------------------------
    # 公開 UI
    # ---------------------------------------------------------
    def render(self) -> None:
        state: Dict[str, Any] = st.session_state[self.session_key]
        current_scene = state.get("current_scene", "TOWN")
        pending_scene = state.get("pending_scene", current_scene)

        scene_mods: Dict[str, Dict[str, float]] = st.session_state.get(
            "scene_emotion_mods", self._create_default_mods()
        )

        st.markdown("### 現在の場所")
        st.write(f"**{self.SCENES[current_scene]['label']}**")

        st.markdown("### 移動先を選ぶ")

        # シーン選択セレクタ
        options = list(self.SCENES.keys())
        if pending_scene not in options:
            pending_scene = current_scene

        idx_default = options.index(pending_scene)

        selected_scene = st.selectbox(
            "移動先シーン",
            options=options,
            index=idx_default,
            format_func=lambda sid: self.SCENES[sid]["label"],
        )

        # pending_scene 更新
        state["pending_scene"] = selected_scene
        st.session_state[self.session_key] = state

        meta = self.SCENES[selected_scene]
        st.info(meta.get("description", ""))

        # -----------------------------------
        # このシーンの感情ボーナス/ペナルティ設定
        # -----------------------------------
        st.markdown("#### このシーンの感情ボーナス / ペナルティ")

        mods_for_scene = scene_mods.get(selected_scene, {k: 0.0 for k in self.EMOTION_KEYS})

        cols = st.columns(3)
        for i, key in enumerate(self.EMOTION_KEYS):
            col = cols[i % 3]
            with col:
                label = self._get_emotion_label(key)
                default_val = float(mods_for_scene.get(key, 0.0))
                mods_for_scene[key] = st.slider(
                    label,
                    min_value=-1.0,
                    max_value=1.0,
                    step=0.05,
                    value=default_val,
                    help=self._get_emotion_help(key),
                    key=f"{selected_scene}_{key}",
                )

        # 変更内容を全体 dict に戻す
        scene_mods[selected_scene] = mods_for_scene
        st.session_state["scene_emotion_mods"] = scene_mods

        st.caption("※ 値は -1.0 ～ +1.0 の範囲で、0.0 が『補正なし』です。")

        # -----------------------------------
        # 移動ボタン
        # -----------------------------------
        st.markdown("---")
        if st.button("この場所に移動する", type="primary"):
            state["current_scene"] = selected_scene
            st.session_state[self.session_key] = state
            # メインゲーム側が「シーン変更を検知」できるようなフラグ
            st.session_state["scene_changed"] = True
            st.success(f"{meta['label']} へ移動しました。")

        # -----------------------------------
        # デバッグ表示
        # -----------------------------------
        with st.expander("デバッグ情報（scene_state / scene_emotion_mods）", expanded=False):
            st.json(
                {
                    "scene_state": st.session_state[self.session_key],
                    "scene_emotion_mods": st.session_state.get("scene_emotion_mods", {}),
                }
            )
