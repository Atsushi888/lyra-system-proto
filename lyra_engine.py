# lyra_engine.py
import os
from typing import Any, Dict, List

import streamlit as st

from personas import get_persona
from components import PreflightChecker, DebugPanel, ChatLog, PlayerInput
from conversation_engine import LLMConversation


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

        # UI コンポーネント生成
        self.preflight = PreflightChecker(self.openai_key, self.openrouter_key)
        self.debug_panel = DebugPanel()
        self.chat_log = ChatLog(self.partner_name, self.DISPLAY_LIMIT)
        self.player_input = PlayerInput()

        # LLM 会話エンジン（中で llm_router を呼ぶ）
        self.conversation = LLMConversation(
            system_prompt=self.system_prompt,
            temperature=0.7,
            max_tokens=800,
        )

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

        # 入力欄 → messages に追加 → conversation_engine に丸投げ
        user_text = self.player_input.render()
        if user_text:
            # ユーザー発言を履歴に追加
            self.state["messages"].append(
                {"role": "user", "content": user_text}
            )

            # ===== LLM 呼び出し（すべて conversation_engine 側に委譲） =====
            try:
                reply_text, meta = self.conversation.generate_reply(
                    self.state["messages"]
                )
            except Exception as e:
                reply_text = f"⚠️ 応答生成中にエラーが発生しました: {e}"
                meta = {"route": "error", "exception": str(e)}

            # メタ情報をセッションに保存（DebugPanel 用）
            self.state["llm_meta"] = meta

            # もし空文字だったら、フォールバックメッセージ
            if not reply_text or not reply_text.strip():
                reply_text = (
                    "……うまく返答を生成できなかったみたい。"
                    "もう一度試してくれる？"
                )

            # フローリアの発言として履歴に追加
            self.state["messages"].append(
                {"role": "assistant", "content": reply_text}
            )
            # ===== ここまで LLM 呼び出し =====

        # 最後に会話ログを描画
        messages: List[Dict[str, str]] = self.state.get("messages", [])
        self.chat_log.render(messages)


# ★★★ エントリーポイント ★★★
if __name__ == "__main__":
    engine = LyraEngine()
    engine.render()
