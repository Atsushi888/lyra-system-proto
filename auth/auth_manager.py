from __future__ import annotations
import streamlit as st
from dataclasses import dataclass
from typing import Dict, Any
import bcrypt

try:
    import streamlit_authenticator as stauth
except Exception:
    stauth = None

from auth.roles import Role


@dataclass
class AuthResult:
    name: str | None
    status: bool | None
    username: str | None


class AuthManager:
    """
    1) 可能なら streamlit-authenticator でログインUIを描画
    2) 失敗したら自前フォームにフォールバック（bcrypt 検証）
    """
    def __init__(self) -> None:
        self._secrets = st.secrets
        self._creds  = dict(self._secrets.get("credentials", {}))
        self._cookie = dict(self._secrets.get("cookie", {}))
        self._bypass = bool(self._secrets.get("auth", {}).get("bypass", False))

        self.authenticator = None
        if stauth is not None:
            try:
                self.authenticator = stauth.Authenticate(
                    credentials=self._creds,
                    cookie_name=self._cookie.get("name", "lyra_auth"),
                    key=self._cookie.get("key", "lyra_secret"),
                    cookie_expiry_days=int(self._cookie.get("expiry_days", 30)),
                    auto_hash=False,
                )
            except Exception:
                self.authenticator = None

    # ---------- 公開API ----------
    def render_login(self, location: str = "main") -> AuthResult:
        """ログインフォーム描画（タイトル含む）"""
        st.markdown(
            "<h1 style='text-align:center;'>🔒 Lyra System ログイン</h1>",
            unsafe_allow_html=True,
        )

        if self._bypass:
            st.session_state["authentication_status"] = True
            st.session_state["name"] = "Bypass Admin"
            st.session_state["username"] = list(
                self._creds.get("usernames", {"admin": {}}).keys()
            )[0]
            return AuthResult("Bypass Admin", True, st.session_state["username"])

        if self.authenticator is not None:
            try:
                loc = location if location in ("main", "sidebar", "unrendered") else "main"
                name, auth_status, username = self.authenticator.login(
                    "Lyra System ログイン", loc
                )
                return AuthResult(name, auth_status, username)
            except Exception as e:
                st.warning(
                    "ログインフォームの標準描画に失敗しました。フォールバックに切り替えます。"
                    f"\n\nReason: {type(e).__name__}"
                )

        return self._fallback_login()

    def role(self) -> Role:
        if bool(self._secrets.get("auth", {}).get("bypass", False)):
            return Role.ADMIN
        if not st.session_state.get("authentication_status"):
            return Role.ANON
        uname = st.session_state.get("username")
        meta = self._creds.get("usernames", {}).get(uname, {}) if uname else {}
        r = str(meta.get("role", "USER")).upper()
        return Role.ADMIN if r == "ADMIN" else Role.USER

    def logout(self, location: str = "sidebar"):
        if self.authenticator is not None:
            try:
                loc = location if location in ("main", "sidebar", "unrendered") else "sidebar"
                self.authenticator.logout("Logout", loc)
                return
            except Exception:
                pass
        for k in ("authentication_status", "name", "username"):
            st.session_state.pop(k, None)
        st.success("Logged out.")

    # ---------- 内部：フォールバック ----------
    def _fallback_login(self) -> AuthResult:
        st.info("フォールバック・ログイン（管理者用簡易UI）")
        with st.form("fallback_login", clear_on_submit=False):
            uname = st.text_input("Username / ID")
            pwd   = st.text_input("Password", type="password")
            ok    = st.form_submit_button("Login")

        name = None
        status = None
        if ok:
            user_tbl: Dict[str, Any] = self._creds.get("usernames", {})
            meta = user_tbl.get(uname)
            if not meta:
                st.error("ユーザーが見つかりません。")
                status = False
            else:
                hashed = str(meta.get("password", ""))
                if self._check_bcrypt(pwd, hashed):
                    st.session_state["authentication_status"] = True
                    st.session_state["name"] = meta.get("name") or uname
                    st.session_state["username"] = uname
                    name = st.session_state["name"]
                    status = True
                    st.success("Login success.")
                else:
                    st.error("パスワードが違います。")
                    status = False

        return AuthResult(name, status, st.session_state.get("username"))

    @staticmethod
    def _check_bcrypt(plain: str, hashed: str) -> bool:
        try:
            return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
        except Exception:
            return False
