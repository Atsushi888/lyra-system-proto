from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import streamlit as st
import streamlit_authenticator as stauth
from .roles import Role

@dataclass
class AuthResult:
    status: str
    username: str | None

def _to_plain(obj: Any) -> Any:
    """Secrets系の読み取り専用オブジェクトを、再帰的に純粋なPython型へ変換。"""
    if isinstance(obj, Mapping):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_to_plain(v) for v in obj]
    # 文字列・数値・bool・None はそのまま
    return obj

class AuthManager:
    def __init__(self) -> None:
        # ---- secrets -> 普通の dict へ変換 ----
        raw_credentials = st.secrets.get("credentials", {})
        raw_cookie = st.secrets.get("cookie", {})
        self._credentials = _to_plain(raw_credentials)  # ← これがミュータブル dict

        cookie_name = raw_cookie.get("name", "lyra_auth")
        cookie_key  = raw_cookie.get("key", "change_me")
        cookie_expiry_days = int(raw_cookie.get("expiry_days", 30))

        self.authenticator = stauth.Authenticate(
            credentials=self._credentials,   # ← もう JSON 化しない
            cookie_name=cookie_name,
            key=cookie_key,
            cookie_expiry_days=cookie_expiry_days,
        )

    def render_login(self) -> AuthResult:
        name, auth_status, username = self.authenticator.login(
            "Lyra System ログイン", location="main"
        )
        st.session_state["auth_status"] = auth_status
        st.session_state["auth_user"] = username
        return AuthResult(status=auth_status or "unauthenticated", username=username)

    def role(self) -> Role:
        if st.session_state.get("auth_status") != True:
            return Role.GUEST
        uname = st.session_state.get("auth_user")
        try:
            role_name = (self._credentials["usernames"][uname].get("role") or "USER").upper()
        except Exception:
            role_name = "USER"
        if role_name == "ADMIN":
            return Role.ADMIN
        if role_name in ("DEV", "DEVELOPER"):
            return Role.DEV
        return Role.USER

    def logout_button(self) -> None:
        self.authenticator.logout("Logout", location="sidebar")
