# lyra_engine.py
import os
from typing import Any, Dict, List

import streamlit as st

from personas import get_persona
from components import PreflightChecker, DebugPanel, ChatLog, PlayerInput
from conversation_engine import LLMConversation
from lyra_core import LyraCore

# ページ全体の基本設定
st.set_page_config(page_title="Lyra Engine – フローリア", layout="wide")
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


class LyraEngine:
    MAX_LOG = 500
    DISPLAY_LIMIT = 20000

    def __init__(self):
        # ペルソナの取得（フローリア）
        persona = get_persona("floria_ja")
        self.system_prompt = persona.system_prompt
        self.starter_hint = persona.starter_hint
        self.partner_name = persona.name

        # API キーの取得
        self.openai_key = st.secrets.get(
            "OPENAI_API_KEY",
            os.getenv("OPENAI_API_KEY", ""),
        )
        self.openrouter_key = st.secrets.get(
            "OPENROUTER_API_KEY",
            os.getenv("OPENROUTER_API_KEY", ""),
        )

        if not self.openai_key:
            st.error(
                "OPENAI_API_KEY が未設定です。"
                "Streamlit → Settings → Secrets で設定してください。"
            )
            st.stop()

        # llm_router 用に環境変数へも流しておく（中で os.getenv する前提）
        os.environ["OPENAI_API_KEY"] = self.openai_key
        if self.openrouter_key:
            os.environ["OPENROUTER_API_KEY"] = self.openrouter_key

        # LLM 会話エンジン（中で llm_router を呼ぶ）
        self.conversation = LLMConversation(
            system_prompt=self.system_prompt,
            temperature=0.7,
            max_tokens=800,
        )

        # コア（1ターン会話制御）
        self.core = LyraCore(self.conversation)

        # UI コンポーネント生成
        self.preflight = PreflightChecker(self.openai_key, self.openrouter_key)
        self.debug_panel = DebugPanel()
        self.chat_log = ChatLog(self.partner_name, self.DISPLAY_LIMIT)
        self.player_input = PlayerInput()

        # セッション状態の初期化
        self._init_session_state()

    # セッション初期化
    def _init_session_state(self) -> None:
        # 会話メッセージ
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
            if self.starter_hint:
                st.session_state["messages"].append(
                    {"role": "assistant", "content": self.starter_hint}
                )

        # LLM メタ情報（DebugPanel 用）
        if "llm_meta" not in st.session_state:
            st.session_state["llm_meta"] = None

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
        llm_meta = self.state.get("llm_meta")
        with st.sidebar:
            # DebugPanel 側が meta: Optional[Dict[str, Any]] を受け取る前提
            self.debug_panel.render(llm_meta)

        # 先に会話ログを表示
        messages: List[Dict[str, str]] = self.state.get("messages", [])
        self.chat_log.render(messages)

        # 入力欄 → LyraCore に1ターン処理を委譲
        user_text = self.player_input.render()
        if user_text:
            self.state["messages"], _ = self.core.proceed_turn(user_text, self.state)
            # 入力欄クリアしたいなら、PlayerInput 側で key をリセットする実装にしておく

# ★★★ エントリーポイント ★★★
if __name__ == "__main__":
    engine = LyraEngine()
    engine.render()
