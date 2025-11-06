class LyraEngine:
    MAX_LOG = 500
    DISPLAY_LIMIT = 20000

    def __init__(self):
        persona = get_persona("floria_ja")
        self.system_prompt = persona.system_prompt
        self.starter_hint = persona.starter_hint
        self.partner_name = persona.name

        # APIキーなどの処理は省略

        self.preflight = PreflightChecker(self.openai_key, self.openrouter_key)
        self.debug_panel = DebugPanel()
        self.chat_log = ChatLog(self.partner_name, self.DISPLAY_LIMIT)

        # 💡 セッション初期化を呼ぶ
        self._init_session_state()

    # 🧩 ここにこの関数を追加
    def _init_session_state(self):
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

            # 最初の一言を入れたい場合
            if self.starter_hint:
                st.session_state["messages"].append(
                    {"role": "assistant", "content": self.starter_hint}
                )

    @property
    def state(self):
        return st.session_state

    def render(self):
        st.write("🛫 PreflightChecker.render() 呼び出し前")
        self.preflight.render()
        st.write("🛬 PreflightChecker.render() 呼び出し後")

        with st.sidebar:
            self.debug_panel.render()

        # ✅ messages を渡して ChatLog 描画
        messages = self.state.get("messages", [])
        self.chat_log.render(messages)
