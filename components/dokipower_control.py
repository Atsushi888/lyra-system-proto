# components/dokipower_control.py
from __future__ import annotations

from typing import Dict, Any

import streamlit as st

from actors.emotion_ai import EmotionResult
from actors.emotion.emotion_levels import affection_to_level
from actors.emotion.emotion_state import relationship_stage_from_level


SESSION_KEY = "dokipower_state"

# 初期状態で使う「未適用」マーカー
DEFAULT_MANUAL_CONTROLS = {
    "status": "まだ '適用' ボタンが押されていません。"
}


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
            "doki_level": 0,          # 0〜4
            "relationship_level": 20,  # 長期的な関係の深さ（0〜100）
            "masking_level": 30,       # ばけばけ度（0〜100）
            "environment": "alone",    # "alone" / "with_others"
        }

    # emotion_manual_controls もここで最低限の初期化だけしておく
    if "emotion_manual_controls" not in st.session_state:
        st.session_state["emotion_manual_controls"] = dict(DEFAULT_MANUAL_CONTROLS)

    return st.session_state[SESSION_KEY]


class DokiPowerController:
    """
    ドキドキ💓パワーと EmotionResult ＋長期関係度／ばけばけ度を
    手動調整するためのコントローラ（デバッグ用）。
    """

    def __init__(self, *, session_key: str = SESSION_KEY) -> None:
        self.session_key = session_key

    @property
    def state(self) -> Dict[str, Any]:
        return _get_state()

    def _set_state(self, data: Dict[str, Any]) -> None:
        st.session_state[self.session_key] = dict(data)

    # ==========================================================
    # UI 本体
    # ==========================================================
    def render(self) -> None:
        state = self.state

        # ===== 基本感情 =====
        st.subheader("基本感情値")

        mode = st.selectbox(
            "mode",
            options=["normal", "erotic", "debate"],
            index=["normal", "erotic", "debate"].index(
                state.get("mode", "normal")
                if state.get("mode", "normal") in ["normal", "erotic", "debate"]
                else "normal"
            ),
        )

        affection = st.slider(
            "affection（好意）",
            0.0,
            1.0,
            float(state.get("affection", 0.5)),
            step=0.05,
        )

        arousal = st.slider(
            "arousal（感情の高まり）",
            0.0,
            1.0,
            float(state.get("arousal", 0.3)),
            step=0.05,
        )

        # ===== 長期関係度 & ばけばけ度 =====
        st.subheader("長期関係度 & ばけばけ度")

        relationship_level = st.slider(
            "relationship_level（長期的な関係の深さ・0〜100）",
            0,
            100,
            int(state.get("relationship_level", 20)),
            help=(
                "0 = ほぼ他人 / 20〜39 = 先輩後輩・友達 "
                "/ 40〜59 = 両想い手前〜安定しつつある恋人候補 "
                "/ 60〜79 = 事実上の恋人 "
                "/ 80〜100 = 夫婦同然・家族レベル"
            ),
        )

        masking_level = st.slider(
            "masking_level（ばけばけ度：感情を“平静”に見せるうまさ・0〜100）",
            0,
            100,
            int(state.get("masking_level", 30)),
            help=(
                "0 = 感情ダダ漏れ / 20〜39 = やや表に出やすい "
                "/ 40〜59 = そこそこ隠せる "
                "/ 60〜79 = よほどのことがなければ表に出ない "
                "/ 80〜100 = かなりの役者。内心は悟らせない。"
            ),
        )

        # ===== ドキドキパワー =====
        st.subheader("ドキドキ💓パワー（その場の高揚感）")

        doki_power = st.slider(
            "doki_power（0〜100：目の前にしたときの一時的な胸の高鳴り）",
            0.0,
            100.0,
            float(state.get("doki_power", 0.0)),
            step=1.0,
        )

        # 自動レベル判定（暫定）
        auto_level = 0
        if doki_power >= 85:
            auto_level = 4
        elif doki_power >= 60:
            auto_level = 3
        elif doki_power >= 40:
            auto_level = 2
        elif doki_power >= 20:
            auto_level = 1

        st.caption(
            f"自動レベル判定（暫定）: {auto_level} "
            "（20/40/60/85 付近で 1/2/3/4）"
        )

        doki_level = st.slider(
            "doki_level（0〜4：段階インデックス・手動上書き可）",
            0,
            4,
            int(state.get("doki_level", auto_level)),
        )

        # ===== 周囲の状況（party_mode 相当） =====
        env_default = state.get("environment", "alone")
        if env_default not in ("alone", "with_others"):
            env_default = "alone"

        environment = st.radio(
            "周囲の状況",
            options=["alone", "with_others"],
            index=["alone", "with_others"].index(env_default),
            format_func=lambda k: "二人きり (alone)" if k == "alone" else "他にも人がいる (with_others)",
            horizontal=True,
        )

        # ===== EmotionResult を構築（スライダー値ベースのプレビュー） =====
        emo = EmotionResult(
            mode=mode,
            affection=affection,
            arousal=arousal,
            doki_power=doki_power,
            doki_level=doki_level,
            relationship_level=float(relationship_level),
            masking_degree=float(masking_level) / 100.0,
        )

        # relationship_level → stage / label を反映
        stage = relationship_stage_from_level(float(relationship_level))
        stage_to_label = {
            "acquaintance": "neutral",
            "friendly": "friend",
            "close_friends": "close_friend",
            "dating": "lover",
            "soulmate": "soulmate",
        }
        emo.relationship_stage = stage
        emo.relationship_label = stage_to_label.get(stage, "neutral")

        st.markdown("---")
        st.subheader("現在の EmotionResult（スライダー値プレビュー）")
        st.json(emo.to_dict())

        # ドキドキ補正後の好感度＆レベル表示
        aff_with_doki = getattr(emo, "affection_with_doki", emo.affection)
        level = affection_to_level(aff_with_doki)

        st.info(
            f"affection_with_doki = {aff_with_doki:.3f} "
            "（ドキドキ💓補正後の実効好感度）"
        )

        level_label_map = {
            "low": "LOW（まだ憧れ段階）",
            "mid": "MID（かなり仲良し）",
            "high": "HIGH（ほぼ両想い）",
            "extreme": "EXTREME（婚前レベル）",
        }
        st.write("現在の好感度レベル:", level_label_map.get(level, level))

        st.markdown("---")

        # ===== コントローラ自身の状況確認 =====
        st.subheader("環境ステータス（このコントローラ固有の情報）")

        env_label = (
            "二人きり (alone)" if environment == "alone" else "他にも人がいる (with_others)"
        )

        st.markdown(f"- 周囲の状況: {env_label}")
        st.markdown(f"- relationship_level（プレビュー）: **{relationship_level}**")
        st.markdown(
            f"- masking_level（スライダー値） = **{masking_level}** → "
            f"EmotionResult.masking_degree = **{emo.masking_degree:.2f}**"
        )

        # emotion_manual_controls の中身を確認
        with st.expander("適用済み emotion_manual_controls の中身を見る", expanded=False):
            manual = st.session_state.get("emotion_manual_controls", DEFAULT_MANUAL_CONTROLS)
            if isinstance(manual, dict) and "status" not in manual:
                # status キーが無ければ「適用済み」とみなす
                st.json(manual)
            else:
                st.json(DEFAULT_MANUAL_CONTROLS)

        st.markdown("---")

        # ===== 適用／リセット =====
        col_apply, col_reset = st.columns(2)

        with col_apply:
            if st.button("✅ この値を Mixer デバッグ用に適用", type="primary"):
                new_state = {
                    "mode": mode,
                    "affection": affection,
                    "arousal": arousal,
                    "doki_power": doki_power,
                    "doki_level": doki_level,
                    "relationship_level": relationship_level,
                    "masking_level": masking_level,
                    "environment": environment,
                }
                self._set_state(new_state)

                # MixerAI などが読む用の EmotionResult
                st.session_state["mixer_debug_emotion"] = emo.to_dict()

                # 手動パラメータ本体（status キーは付けない）
                st.session_state["emotion_manual_controls"] = {
                    "relationship_level": int(relationship_level),
                    "doki_power": float(doki_power),
                    "masking_level": int(masking_level),
                    "environment": environment,
                }

                st.success(
                    "EmotionResult を session_state['mixer_debug_emotion'] に、"
                    "手動パラメータを session_state['emotion_manual_controls'] に保存しました。"
                )

        with col_reset:
            if st.button("🔁 初期値にリセット"):
                init_state = {
                    "mode": "normal",
                    "affection": 0.5,
                    "arousal": 0.3,
                    "doki_power": 0.0,
                    "doki_level": 0,
                    "relationship_level": 20,
                    "masking_level": 30,
                    "environment": "alone",
                }
                self._set_state(init_state)

                # 未適用マーカーに戻す
                st.session_state["emotion_manual_controls"] = dict(DEFAULT_MANUAL_CONTROLS)

                st.info("ドキドキ💓パワー / 感情値 / 手動パラメータを初期状態に戻しました。")
