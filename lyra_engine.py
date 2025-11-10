# lyra_engine.py — Lyra Engine main entrypoint

import os
from typing import Any, Dict, List

import streamlit as st

from personas.persona_floria_ja import get_personaf
from components import PreflightChecker, DebugPanel, ChatLog, PlayerInput
from deliveration.multi_ai_response import MultiAIResponse
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
        # ペルソナの取得（現時点ではフローリア固定）
        persona = get_persona()
        self.system_prompt = persona.system_prompt
        self.starter_hint = persona.starter_hint
        self.partner_name = persona.name
        self.style_hint = persona.style_hint  # ← ペルソナ側の文体指針

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

        # ===== LLM 会話エンジン（中で llm_router を呼ぶ） =====
        self.conversation = LLMConversation(
            system_prompt=self.system_prompt,
            temperature=0.7,
            max_tokens=800,
            style_hint=self.style_hint,
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

    # ===== セッション初期化 =====
    def _init_session_state(self) -> None:
        # 会話ログ
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
            if self.starter_hint:
                st.session_state["messages"].append(
                    {"role": "assistant", "content": self.starter_hint}
                )

        # LLM メタ情報
        if "llm_meta" not in st.session_state:
            st.session_state["llm_meta"] = None

        # 裏画面 ON/OFF フラグ
        if "debug_mode" not in st.session_state:
            st.session_state["debug_mode"] = False

    @property
    def state(self):
        return st.session_state

    # ===== メインレンダリング =====
    def render(self) -> None:
        # Preflight（キー診断）
        self.preflight.render()

        # サイドバー（裏ビュー切り替え + デバッグパネル）
        with st.sidebar:
            # 裏画面トグルボタン
            if st.button("🧠 マルチAI裏ビュー切替"):
                st.session_state["debug_mode"] = not st.session_state["debug_mode"]

            mode_label = "裏画面 ON" if st.session_state["debug_mode"] else "裏画面 OFF"
            st.caption(f"現在: {mode_label}")

            # 既存のデバッグパネル（llm_meta の簡易表示など）
            # llm_meta = self.state.get("llm_meta")
            # self.debug_panel.render(llm_meta)

        # 表 / 裏 切り替え
        if st.session_state["debug_mode"]:
            self.render_backstage()
        else:
            self.render_front()

    # ===== 表画面（プレイヤー用：従来のLyra画面） =====
    def render_front(self) -> None:
        """いつものフローリア会話画面。"""

        # ① 現在の会話ログを表示
        messages: List[Dict[str, str]] = self.state.get("messages", [])
        self.chat_log.render(messages)

        # ② プレイヤー入力欄
        user_text = self.player_input.render()

        if user_text:
            with st.spinner("フローリアが返事を考えています…"):
                updated_messages, meta = self.core.proceed_turn(
                    user_text,
                    self.state,
                )

            # 整形後のメッセージを state に反映
            self.state["messages"] = updated_messages
            self.state["llm_meta"] = meta

            # （必要ならスクロール用のフラグもここで立てる）
            self.state["scroll_to_input"] = True

            st.rerun()

    # ===== 裏画面（開発者用：マルチAIリプライ可視化） =====
    def render_backstage(self) -> None:
        """裏画面：マルチAIリプライ可視化ビュー。"""

        st.markdown("## 🎭 Lyra Backstage – Multi AI Response")

        llm_meta: Dict[str, Any] | None = self.state.get("llm_meta")

        viewer = MultiAIResponse(title="マルチAIレスポンス（デバッグ）")
        viewer.render(llm_meta)


# ===== エントリーポイント =====
if __name__ == "__main__":
    engine = LyraEngine()
    engine.render()
