# lyra_engine.py
import os
import json
import time
from typing import Any, Dict, List

import streamlit as st

from personas import get_persona
from llm_router import call_with_fallback
#from components import PreflightChecker, DebugPanel, ChatLog, PlayerInput
import components.preflight as preflight
import components.debug_panel as debug_panel
import components.chat_log as chat_log
import components.player_input as player_input

# ページ全体の基本設定
st.set_page_config(page_title="Lyra Engine – フローリア", layout="wide")
st.markdown("""
<style>
.chat-bubble {
    border: 1px solid #ccc;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 6px 0;
    background-color: #f9f9f9;
}
.chat-bubble.user {
    border-color: #66aaff;
    background-color: #e8f2ff;
}
.chat-bubble.assistant {
    border-color: #999;
    background-color: #f2f2f2;
}
</style>
""", unsafe_allow_html=True)



class LyraEngine:
    MAX_LOG = 500
    DISPLAY_LIMIT = 20000

    def __init__(self):
        persona = get_persona("floria_ja")
        self.system_prompt = persona.system_prompt
        self.starter_hint = persona.starter_hint
        self.partner_name = persona.name

        # API キーの取得
        self.openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        self.openrouter_key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", ""))

        if not self.openai_key:
            st.error("OPENAI_API_KEY が未設定です。Streamlit → Settings → Secrets で設定してください。")
            st.stop()

        os.environ["OPENAI_API_KEY"] = self.openai_key
        if self.openrouter_key:
            os.environ["OPENROUTER_API_KEY"] = self.openrouter_key

        # UI コンポーネント生成
        # self.preflight = PreflightChecker(self.openai_key, self.openrouter_key)
        # self.debug_panel = DebugPanel()
        # self.chat_log = ChatLog(self.partner_name, self.DISPLAY_LIMIT)
        # self.player_input = PlayerInput()   # ← ここ追加

        # ★ セッション状態の初期化
        self._init_session_state()

    # ★★★ ここは必ず class の中（__init__ と同じインデント）に置く ★★★
    def _init_session_state(self) -> None:
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
            if self.starter_hint:
                st.session_state["messages"].append(
                    {"role": "assistant", "content": self.starter_hint}
                )

    @property
    def state(self):
        return st.session_state

    def render(self) -> None:
        # ここまで来ているかの確認
        st.write("✅ Lyra Engine 起動テスト：render() まで来てます。")

        # Preflight（キー診断）
        st.write("🛫 PreflightChecker.render() 呼び出し前")
        self.preflight.render()
        st.write("🛬 PreflightChecker.render() 呼び出し後")

        # デバッグパネル（サイドバー）
        with st.sidebar:
            self.debug_panel.render()

        # 会話ログ
        messages: List[Dict[str, str]] = self.state.get("messages", [])
        self.chat_log.render(messages)
        
        # 入力欄
        user_text = self.player_input.render()
        if user_text:
            st.session_state["messages"].append({"role": "user", "content": user_text})
            st.session_state["messages"].append(
                {"role": "assistant", "content": "（まだ応答生成ロジック未実装）"}
            )
            st.experimental_rerun()


# ★★★ エントリーポイント ★★★
if __name__ == "__main__":
    engine = LyraEngine()
    engine.render()
