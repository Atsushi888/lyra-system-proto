# actors/actor.py

from __future__ import annotations
from typing import List, Dict, Any
import os

import streamlit as st

from actors.answer_talker import AnswerTalker

# 環境変数 LYRA_DEBUG=1 で簡易デバッグログを有効化
LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"


class Actor:
    def __init__(self, name: str, persona: Any) -> None:
        """
        name  : "floria" / "riseria" など、キャラクターID用途
        persona : Persona クラスインスタンス（フローリア用でもリセリア用でもOK）
        """
        self.name = name
        self.persona = persona

        # LLMRouterはもう使わない。AnswerTalker内部で自動的にLLMManager/Routerへ接続
        self.answer_talker = AnswerTalker(persona=self.persona)

    # --------------------------------------------------
    # 会話ログからプレイヤーの最新発言を抽出
    # --------------------------------------------------
    @staticmethod
    def _extract_latest_user_text(conversation_log: List[Dict[str, Any]]) -> str:
        """
        conversation_log から、最後のプレイヤー発話を取り出す。

        ログのフォーマット差異に耐えるため、
        - role == "player" / "user" を優先
        - テキストキーは "content" 優先 → なければ "text"
        として拾う。
        """
        for entry in reversed(conversation_log):
            role = entry.get("role") or entry.get("speaker") or ""
            if role in ("player", "user"):
                return entry.get("content") or entry.get("text") or ""
        return ""

    # --------------------------------------------------
    # Persona による messages 構築
    # --------------------------------------------------
    def _build_messages(self, conversation_log: List[Dict[str, Any]], user_text: str) -> List[Dict[str, str]]:
        """
        Persona 側に build_messages() があればそれを呼び出し、
        失敗したときは user_text だけを使ったフォールバックを行う。
        """
        messages: List[Dict[str, str]] = []

        if hasattr(self.persona, "build_messages"):
            try:
                # 新仕様: conversation_log まるごと渡す
                messages = self.persona.build_messages(conversation_log)
                if LYRA_DEBUG:
                    st.write(
                        f"[DEBUG:Actor.{self.name}] persona.build_messages(conversation_log) "
                        f"→ len={len(messages)}"
                    )
                return messages
            except TypeError:
                # 旧仕様: user_text だけを受け取る版へのフォールバック
                try:
                    messages = self.persona.build_messages(user_text)
                    if LYRA_DEBUG:
                        st.write(
                            f"[DEBUG:Actor.{self.name}] persona.build_messages(user_text) "
                            f"(fallback) → len={len(messages)}"
                        )
                    return messages
                except Exception as e:
                    if LYRA_DEBUG:
                        st.write(
                            f"[DEBUG:Actor.{self.name}] persona.build_messages error: {e}"
                        )

        # ---- フォールバック（最低限の system + user）----
        if LYRA_DEBUG:
            st.write(
                f"[DEBUG:Actor.{self.name}] build_messages fallback path (simple system+user)."
            )

        system_prompt = ""
        if hasattr(self.persona, "get_system_prompt"):
            try:
                system_prompt = self.persona.get_system_prompt()
            except Exception:
                system_prompt = ""

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_text:
            messages.append({"role": "user", "content": user_text})

        return messages

    # --------------------------------------------------
    # メイン入口
    # --------------------------------------------------
    def speak(self, conversation_log: List[Dict[str, Any]]) -> str:
        """
        会話システムから呼ばれるメイン関数。

        conversation_log:
            CouncilManager 側で管理している [ { role, text/content, ... }, ... ]
        """
        if LYRA_DEBUG:
            st.write(
                f"[DEBUG:Actor.{self.name}] speak() called. "
                f"log_len={len(conversation_log)}"
            )

        # プレイヤーの最新発言を取得
        user_text = self._extract_latest_user_text(conversation_log)

        if LYRA_DEBUG:
            st.write(
                f"[DEBUG:Actor.{self.name}] latest user_text = {repr(user_text)[:120]}"
            )

        # Persona に messages を作らせる（新旧両方のシグネチャに対応）
        messages = self._build_messages(conversation_log, user_text)

        if LYRA_DEBUG:
            st.write(
                f"[DEBUG:Actor.{self.name}] messages built. len={len(messages)}"
            )

        # AnswerTalker によるLLMパイプライン処理
        final_reply = self.answer_talker.speak(messages, user_text=user_text)

        if LYRA_DEBUG:
            st.write(
                f"[DEBUG:Actor.{self.name}] final_reply len={len(final_reply)}, "
                f"preview={repr(final_reply[:200])}"
            )

        return final_reply
