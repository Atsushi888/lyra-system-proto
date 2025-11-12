# auth/auth_manager.py

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional, Tuple

import streamlit as st
import streamlit_authenticator as stauth

from .roles import Role


def _to_plain_dict(obj) -> Dict:
    """
    st.secrets は Secrets オブジェクト（辞書ライク）なので、
    streamlit_authenticator に渡す前に通常の dict に落とす。
    ネストされた要素もできるだけ dict 化して返す。
    """
    if hasattr(obj, "to_dict"):
        return obj.to_dict()  # type: ignore[attr-defined]
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    return deepcopy(obj)


class AuthManager:
    """
    Streamlit 認証ラッパー。
    - st.secrets から資格情報（credentials / cookie）を読み取り
    - streamlit_authenticator を初期化
    - 役割（Role）を取得できるようにする
    """

    def __init__(self) -> None:
        if "credentials" not in st.secrets:
            raise RuntimeError("st.secrets['credentials'] が見つかりません。secrets.toml を設定してください。")

        self.credentials: Dict = _to_plain_dict(st.secrets["credentials"])
        self.cookie: Dict = _to_plain_dict(st.secrets.get("cookie", {}))

        self.authenticator = stauth.Authenticate(
            credentials=self.credentials,
            cookie_name=self.cookie.get("name", "lyra_auth"),
            key=self.cookie.get("key", "change_me"),
            cookie_expiry_days=int(self.cookie.get("expiry_days", 30)),
        )

    # ------------------------------
    # ロール判定
    # ------------------------------
    def _role_from_username(self, username: Optional[str]) -> Role:
        if not username:
            return Role.GUEST

        rec = self.credentials.get("usernames", {}).get(username, {})
        key = str(rec.get("role", "USER")).upper()

        mapping = {
            "ADMIN": Role.ADMIN,
            "USER": Role.USER,
            "DEV": Role.DEV,
            "DEVELOPER": Role.DEV,
            "GUEST": Role.GUEST,
        }
        return mapping.get(key, Role.USER)

    def role(self) -> Role:
        """現在ログイン中ユーザーの Role（未ログインなら GUEST）"""
        if not st.session_state.get("authentication_status"):
            return Role.GUEST
        return self._role_from_username(st.session_state.get("username"))

    def current_user_display(self) -> str:
        """表示名（未ログインなら空）"""
        if not st.session_state.get("authentication_status"):
            return ""
        return str(st.session_state.get("name") or st.session_state.get("username") or "")

    # ------------------------------
    # 画面部品
    # ------------------------------
    def render_login(self, location: str = "main") -> None:
        """
        ログインフォームを描画して状態を session_state に反映する。
        streamlit_authenticator の API 差異に備えてフォールバック実装。
        """
        try:
            name, auth_status, username = self.authenticator.login(
                location=location,
                key="lyra_auth_login",
                fields={"Form name": "Lyra System ログイン"},
            )
        except TypeError:
            # 古いバージョンの引数シグネチャ
            name, auth_status, username = self.authenticator.login(
                "Lyra System ログイン", location
            )

        st.session_state["authentication_status"] = auth_status
        st.session_state["name"] = name
        st.session_state["username"] = username

        if auth_status is True:
            st.success(f"Logged in: {name or username}")
        elif auth_status is False:
            st.error("メール / パスワードが違います。")
        else:
            st.info("メール / パスワードを入力してください。")

    def render_logout(self, location: str = "sidebar") -> None:
        """ログアウトボタンを表示"""
        try:
            self.authenticator.logout(location=location, key="lyra_auth_logout")
        except TypeError:
            self.authenticator.logout("Logout", location)

    # ------------------------------
    # ガード
    # ------------------------------
    def require(self, min_role: Role) -> Role:
        """
        画面の先頭で呼び出し、min_role 未満ならログイン画面を出して停止する。
        戻り値は現在の Role。
        """
        r = self.role()
        if r < min_role:
            self.render_login(location="main")
            st.stop()
        return r
