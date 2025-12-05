# components/dokipower_control.py
from __future__ import annotations

from typing import Dict, Any

import streamlit as st

from actors.emotion_ai import EmotionResult


SESSION_KEY = "dokipower_state"


def _get_state() -> Dict[str, Any]:
    """
    サイドウインドウ内のスライダー状態を session_state に保持。
    """
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = {
            "mode": "normal",
            "affection": 0.5,
            "arousal": 0.3,
            "doki_power": 0.0,
            "doki_level": 0,
        }
    return st.session_state[SESSION_KEY]


class DokiPowerController:
    """
    ドキドキ💓パワーと EmotionResult を手動調整するためのコントローラ。

    - affection / arousal / doki_power / doki_level をスライダーで操作
    - 適用すると EmotionResult を session_state["mixer_debug_emotion"] に書き込み
      → MixerAI などがここを読めば、即「効き目」を確認できる。
    """

    def __init__(self, *, session_key: str = SESSION_KEY) -> None:
        self.session_key = session_key

    @property
    def state(self) -> Dict[str, Any]:
        return _get_state()

    def _set_state(self, data: Dict[str, Any]) -> None:
        st.session_state[self.session_key] = dict(data)

    def render(self) -> None:
        state = self.state

        # ===== 基本感情 =====
        st.subheader("基本感情値")

        col1, col2 = st.columns(2)
        with col1:
            mode = st.selectbox(
                "mode",
                options=["normal", "erotic", "debate"],
                index=["normal", "erotic", "debate"].index(
                    state.get("mode", "normal")
                    if state.get("mode", "normal") in ["normal", "erotic", "debate"]
                    else "normal"
                ),
            )
        with col2:
            affection = st.slider(
                "affection（好意）",
                0.0, 1.0,
                float(state.get("affection", 0.5)),
                step=0.05,
            )

        arousal = st.slider(
            "arousal（感情の高まり）",
            0.0, 1.0,
            float(state.get("arousal", 0.3)),
            step=0.05,
        )

        # ===== ドキドキパワー =====
        st.subheader("ドキドキ💓パワー")

        doki_power = st.slider(
            "doki_power（0〜100）",
            0.0, 100.0,
            float(state.get("doki_power", 0.0)),
            step=1.0,
        )

        # しきい値から自動レベル判定（手動で上書き可）
        auto_level = 0
        if doki_power >= 80:
            auto_level = 3
        elif doki_power >= 50:
            auto_level = 2
        elif doki_power >= 25:
            auto_level = 1

        st.caption(f"自動レベル判定（暫定）: {auto_level}（25/50/80 で 1/2/3）")

        doki_level = st.slider(
            "doki_level（段階インデックス・手動上書き可）",
            0, 3,
            int(state.get("doki_level", auto_level)),
        )

        # ===== EmotionResult を構築 =====
        emo = EmotionResult(
            mode=mode,
            affection=affection,
            arousal=arousal,
            doki_power=doki_power,
            doki_level=doki_level,
        )

        st.markdown("---")
        st.subheader("現在の EmotionResult（プレビュー）")
        st.json(emo.to_dict())

        st.info(
            f"affection_with_doki = {emo.affection_with_doki:.3f} "
            "（ドキドキ💓補正後の実効好感度）"
        )

        # ===== 適用／リセット =====
        st.markdown("---")
        col_apply, col_reset = st.columns(2)

        with col_apply:
            if st.button("✅ この値を Mixer デバッグ用に適用", type="primary"):
                new_state = {
                    "mode": mode,
                    "affection": affection,
                    "arousal": arousal,
                    "doki_power": doki_power,
                    "doki_level": doki_level,
                }
                self._set_state(new_state)

                # MixerAI などが読む用のキー
                st.session_state["mixer_debug_emotion"] = emo.to_dict()
                st.success(
                    "EmotionResult を session_state['mixer_debug_emotion'] に保存しました。"
                )

        with col_reset:
            if st.button("🔁 リセット（初期値に戻す）"):
                init_state = {
                    "mode": "normal",
                    "affection": 0.5,
                    "arousal": 0.3,
                    "doki_power": 0.0,
                    "doki_level": 0,
                }
                self._set_state(init_state)
                st.info("ドキドキ💓パワーと感情値を初期状態に戻しました。")
