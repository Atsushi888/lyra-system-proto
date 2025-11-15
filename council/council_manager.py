# council/council_manager.py

from __future__ import annotations
from typing import List, Dict, Any

import streamlit as st

from actors.actor import Actor
from personas.persona_floria_ja import Persona, get_persona


class CouncilManager:
    """
    会談システムのロジック側（β）。
    Actor ベースで応答を生成しつつ、状態はすべて session_state に保持する。
    """

    PREFIX = "council_"  # session_state 用キーのプレフィックス

    def __init__(self) -> None:
        self.ss = st.session_state
        self._ensure_state()

        # フローリアのペルソナを取得
        floria_persona: Persona = get_persona()

        # Actor インスタンス（LLM ラッパ）
        # Actor 自体は毎フレーム作り直しても、状態は session_state 側で持つので OK
        self.actors: Dict[str, Actor] = {
            "floria": Actor("フローリア", floria_persona)
        }

    # ===== 内部ヘルパ =====
    def _key(self, name: str) -> str:
        return f"{self.PREFIX}{name}"

    def _ensure_state(self) -> None:
        """初期状態がなければ作る。"""
        if self._key("log") not in self.ss:
            self.ss[self._key("log")] = []  # List[Dict[str, str]]
        if self._key("state") not in self.ss:
            self.ss[self._key("state")] = {
                "round": 1,
                "speaker": "player",   # 現在のターンの話者（今は player 固定）
                "mode": "ongoing",     # "ongoing" / "ended" など将来拡張
            }

    # プロパティで見やすく
    @property
    def conversation_log(self) -> List[Dict[str, str]]:
        return self.ss[self._key("log")]

    @property
    def state(self) -> Dict[str, Any]:
        return self.ss[self._key("state")]

    # ===== 公開 API =====
    def reset(self) -> None:
        """会談をリセットして最初からやり直し。"""
        self.ss[self._key("log")] = []
        self.ss[self._key("state")] = {
            "round": 1,
            "speaker": "player",
            "mode": "ongoing",
        }

    def _append_log(self, role: str, content: str) -> None:
        """
        会話ログに 1 件追加。
        改行は Markdown で効くように "  \n" に変換する。
        """
        safe_text = content.replace("\n", "  \n")
        self.conversation_log.append(
            {
                "role": role,
                "content": safe_text,
            }
        )

    def proceed(self, user_text: str) -> str:
        """
        プレイヤー発言 → フローリア応答までをまとめて処理する。
        戻り値: フローリアの生返答（整形前のテキスト）
        """

        if self.state.get("mode") != "ongoing":
            # 将来、会談終了後などに備えて一応ガード
            return ""

        # 1) プレイヤー発言をログに積む
        self._append_log("player", user_text)

        # 2) フローリア（Actor）に会話ログを渡して返事をもらう
        ai_reply = ""
        actor = self.actors.get("floria")
        if actor is not None:
            ai_reply = actor.speak(self._build_llm_history())

            # 3) フローリアの返答もログに積む
            self._append_log("floria", ai_reply)

        # 4) ラウンドを進める
        self.state["round"] = int(self.state.get("round", 1)) + 1

        # 「話者が player 一人だけに見える」問題への、とりあえずの軽い対処として
        # 「最後に発言した人」を記録しておく（今はフローリア）
        self.state["last_speaker"] = "floria"

        return ai_reply

    # ===== LLM 用履歴変換 =====
    def _build_llm_history(self) -> List[Dict[str, str]]:
        """
        Actor.speak() に渡すための LLM 風 history を構築する。
        今は簡易的に:
          - player → {"role": "user", "content": ...}
          - floria → {"role": "assistant", "content": ...}
        """
        history: List[Dict[str, str]] = []
        for entry in self.conversation_log:
            role = entry.get("role")
            text = entry.get("content", "").replace("  \n", "\n")  # LLM 向けに戻す

            if role == "player":
                history.append({"role": "user", "content": text})
            elif role == "floria":
                history.append({"role": "assistant", "content": text})
            else:
                # 将来 system / 他参加者など
                history.append({"role": "system", "content": text})

        return history
