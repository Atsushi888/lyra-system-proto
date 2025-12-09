from __future__ import annotations

from typing import Any, Dict

import streamlit as st


SESSION_KEY = "user_settings"


def _get_default_settings() -> Dict[str, Any]:
    """
    UserSettings のデフォルト値。
    必要になったらここに項目を追加していく。
    """
    return {
        "player_name": "アツシ",
        # "auto" / "short" / "normal" / "long" / "story"
        # ★ デフォルトを story に変更
        "reply_length_mode": "story",
    }


def _ensure_state() -> Dict[str, Any]:
    """
    session_state 内に user_settings が無ければ初期化して返す。
    あわせて、よく使う値はトップレベルにもミラーしておく。
    """
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = _get_default_settings()
    else:
        # 既存に新しいキーが増えたときのためにマージ
        current = dict(_get_default_settings())
        current.update(st.session_state[SESSION_KEY] or {})
        st.session_state[SESSION_KEY] = current

    settings: Dict[str, Any] = st.session_state[SESSION_KEY]

    # 他モジュールが直接参照しやすいよう、トップレベルにも置いておく
    st.session_state.setdefault("player_name", settings.get("player_name", "アツシ"))
    st.session_state.setdefault(
        "reply_length_mode", settings.get("reply_length_mode", "story")
    )

    return settings


class UserSettings:
    """
    ユーザー設定（プレイヤー名・発話長さモードなど）を扱うコンポーネント。

    - 設定値は session_state["user_settings"] にまとめて保存
    - 便利のため、よく使う値はトップレベルキーにもミラーする：
        - session_state["player_name"]
        - session_state["reply_length_mode"]
    """

    def __init__(self, *, session_key: str = SESSION_KEY) -> None:
        self.session_key = session_key
        _ensure_state()  # 初期化だけしておく

    @property
    def settings(self) -> Dict[str, Any]:
        return _ensure_state()

    def _save_settings(self, new_settings: Dict[str, Any]) -> None:
        # メインの設定
        st.session_state[self.session_key] = dict(new_settings)

        # よく使うキーはトップレベルにもコピー
        player_name = new_settings.get("player_name") or "アツシ"
        reply_length_mode = new_settings.get("reply_length_mode") or "story"

        st.session_state["player_name"] = player_name
        st.session_state["reply_length_mode"] = reply_length_mode

    # --------- パブリックな取得ヘルパ（他モジュールから使う想定） ---------

    def get_player_name(self) -> str:
        return self.settings.get("player_name", "アツシ")

    def get_reply_length_mode(self) -> str:
        return self.settings.get("reply_length_mode", "story")

    # --------- UI レンダリング ---------

    def render(self) -> None:
        st.subheader("ユーザー基本設定")

        settings = self.settings

        # プレイヤー名
        player_name = st.text_input(
            "プレイヤー名（ゲーム内で呼ばれる名前）",
            value=settings.get("player_name", "アツシ"),
            max_chars=32,
            help="例：アツシ / トーマ / Atsushi など。Persona の {PLAYER_NAME} に反映されます（新しい会話から有効）。",
        )

        st.markdown("---")
        st.subheader("会話スタイル設定")

        # 発話の長さモード
        mode_options = ["auto", "short", "normal", "long", "story"]
        mode_labels = {
            "auto": "auto（状況に合わせて自動）",
            "short": "short（1〜2文程度）",
            "normal": "normal（3〜5文程度）",
            "long": "long（5〜8文程度）",
            "story": "story（ミニシーン風で少し長め）",
        }
        # ★ デフォルトも story に
        current_mode = settings.get("reply_length_mode", "story")
        if current_mode not in mode_options:
            current_mode = "story"

        idx = mode_options.index(current_mode)

        reply_length_mode = st.selectbox(
            "リセリアの発話の長さモード",
            options=mode_options,
            format_func=lambda m: mode_labels.get(m, m),
            index=idx,
            help=(
                "short: 1〜2文のコンパクトな返答\n"
                "normal: 通常会話（3〜5文）\n"
                "long: 会話中心・少し長め\n"
                "story: その場の情景も含めたミニシーン風の返答\n"
                "auto: エンジン側の判断に任せます"
            ),
        )

        st.markdown("---")
        col_save, col_reset = st.columns(2)

        with col_save:
            if st.button("✅ 設定を保存", type="primary"):
                new_settings = {
                    "player_name": player_name.strip() or "アツシ",
                    "reply_length_mode": reply_length_mode,
                }
                self._save_settings(new_settings)
                st.success("ユーザー設定を保存しました。")

        with col_reset:
            if st.button("🔁 初期値にリセット"):
                defaults = _get_default_settings()
                self._save_settings(defaults)
                st.info("ユーザー設定を初期状態に戻しました。")

        st.caption(
            "※ プレイヤー名は、新しく生成される Persona から順次反映されます。\n"
            "※ 発話の長さモードは、AnswerTalker / Persona / Composer 側から "
            "system_prompt と最終テキストに反映されます。"
        )
