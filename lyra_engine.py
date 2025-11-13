# lyra_engine.py  — ゲーム本体（進行・会話）だけを担当
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import os
import streamlit as st

from personas.persona_floria_ja import get_persona
from components import PreflightChecker, ChatLog, PlayerInput
from conversation_engine import LLMConversation
from lyra_core import LyraCore

class LyraEngine:
    MAX_LOG = 500
    DISPLAY_LIMIT = 20000

    def __init__(self) -> None:
        # ペルソナ
        persona = get_persona()
        self.system_prompt = persona.system_prompt
        self.starter_hint = persona.starter_hint
        self.partner_name  = persona.name
        self.style_hint    = persona.style_hint

        # 会話エンジン
        self.conversation = LLMConversation(
            system_prompt=self.system_prompt,
            temperature=st.session_state.get("temp_gpt4o", 0.7),
            max_tokens=st.session_state.get("max_gpt4o", 800),
            style_hint=self.style_hint,
        )
        # 1ターン制御
        self.core = LyraCore(self.conversation)

        # UI 構成物（ゲーム画面用）
        # self.preflight  = PreflightChecker(openai_key, openrouter_key)
        self.chat_log   = ChatLog(self.partner_name, self.DISPLAY_LIMIT)
        self.player_in  = PlayerInput()

        # state 初期化
        self._init_state()

    def _init_state(self) -> None:
        s = st.session_state
        if "messages" not in s:
            s.messages = []
            if self.starter_hint:
                s.messages.append({"role": "assistant", "content": self.starter_hint})
        if "llm_meta" not in s:
            s.llm_meta = None

    @property
    def state(self): return st.session_state

    def render(self) -> None:
        """ゲームの“右側メインビュー”を描画（サイドバーはLyraSystemが持つ）"""
        # Preflight
        # self.preflight.render()

        # ログ表示
        self.chat_log.render(self.state.messages)

        # 入力
        user_text = self.player_in.render()
        if not user_text:
            return

        with st.spinner("フローリアが返事を考えています…"):
            updated_messages, meta = self.core.proceed_turn(user_text, self.state)

        self.state.messages = updated_messages
        self.state.llm_meta = meta
        self.state.scroll_to_input = True
        st.rerun()
