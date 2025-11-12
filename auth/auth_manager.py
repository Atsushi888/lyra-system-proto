# 先頭の import から json はもう不要
# import json  ← これも削除可
from collections.abc import Mapping
import streamlit as st
import streamlit_authenticator as stauth
from .roles import Role

class AuthManager:
    def __init__(self) -> None:
        # --- Secrets → 純粋な dict へ変換するヘルパ ---
        def to_plain(obj):
            if isinstance(obj, Mapping):
                return {k: to_plain(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_plain(v) for v in obj]
            return obj

        # ✨ ここを修正：JSON 経由をやめてプレーン化する
        credentials = to_plain(st.secrets["credentials"])
        cookie_conf = to_plain(st.secrets.get("cookie", {}))
        preauth    = to_plain(st.secrets.get("preauthorized", {}))

        self.authenticator = stauth.Authenticate(
            credentials=credentials,
            cookie_name=cookie_conf.get("name", "lyra_auth"),
            key=cookie_conf.get("key", "lyra_secret"),
            cookie_expiry_days=int(cookie_conf.get("expiry_days", 30)),
            preauthorized=preauth.get("emails", []),
        )

    def render_login(self, location: str = "main"):
        # 許容値に正規化
        loc = location if location in ("main", "sidebar", "unrendered") else "main"
    
        # ✅ location は位置引数で渡す（キーワードにしない）
        name, auth_status, username = self.authenticator.login("Lyra System ログイン", loc)
    
        return type("AuthResult", (), {
            "name": name,
            "status": auth_status,
            "username": username,
        })
    
    def role(self) -> int:
        user = st.session_state.get("username")
        if not user:
            return Role.ANON
        # secrets からロールを取得
        try:
            role_str = st.secrets["credentials"]["usernames"][user]["role"]
        except Exception:
            return Role.USER
        mapping = {"ADMIN": Role.ADMIN, "USER": Role.USER}
        return mapping.get(str(role_str).upper(), Role.USER)
