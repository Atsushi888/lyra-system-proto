from __future__ import annotations
import streamlit as st
import streamlit_authenticator as stauth
from dataclasses import dataclass
from typing import Literal, Tuple, Optional

Location = Literal["main", "sidebar"]

@dataclass
class AuthResult:
    name: Optional[str]
    status: Optional[bool]
    username: Optional[str]

class AuthManager:
    def __init__(self) -> None:
        # secrets.toml からそのまま渡す（auto_hash=False で書込を抑止）
        self._credentials = st.secrets["credentials"]
        self._cookie      = st.secrets["cookie"]

        self.authenticator = stauth.Authenticate(
            credentials=self._credentials,
            cookie_name=self._cookie["name"],
            key=self._cookie["key"],
            cookie_expiry_days=int(self._cookie.get("expiry_days", 30)),
            auto_hash=False,             # ←重要：Secrets書換を抑止
        )

    def login(self, form_name: str="Lyra System ログイン",
              location: Location="main") -> AuthResult:
        # ★ ここで st.form を使わないこと！ 直接呼ぶ
        name, auth_status, username = self.authenticator.login(
            form_name,
            location=location,           # 'main' or 'sidebar'
            key="lyra_login"             # 固定キーでOK
        )
        return AuthResult(name, auth_status, username)

    def logout(self, location: Location="sidebar") -> None:
        self.authenticator.logout(location=location, key="lyra_logout")

    def role(self) -> str:
        # 任意：secretsに保存した役割を返す（未ログイン時は空）
        u = st.session_state.get("username")
        if not u: 
            return ""
        try:
            return self._credentials["usernames"][u].get("role","")
        except Exception:
            return ""
