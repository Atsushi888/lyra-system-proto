# council/council_manager.py

from __future__ import annotations
from typing import List, Dict, Any

from actors.actor import Actor
from personas.persona_floria_ja import Persona


class CouncilManager:
    """
    会談システムのロジック側（β）。
    Actor ベースで応答を生成する最小構成。
    """

    def __init__(self):
        # 会話ログ：List[{role:"", content:""}]
        self.conversation_log: List[Dict[str, str]] = []

        # 一旦フローリアAIだけでよい
        self.actors: Dict[str, Actor] = {
            "floria": Actor("フローリア", Persona())
        }

        self.state = {
            "round": 1,
            "speaker": "player",
            "mode": "ongoing"
        }

    # ---- 会話ログ操作 ----
    def _append_log(self, role: str, content: str):
        self.conversation_log.append({"role": role, "content": content})

    # ---- リセット ----
    def reset(self):
        self.conversation_log.clear()
        self.state = {
            "round": 1,
            "speaker": "player",
            "mode": "ongoing"
        }

    # ---- メイン処理 ----
    def proceed(self, user_text: str):
        """
        プレイヤー発言 → AI応答までをまとめて処理する。
        """
        # プレイヤー発言
        self._append_log("player", user_text)

        # 返事を生成するAI
        actor = self.actors.get("floria")

        if actor:
            ai_reply = actor.speak(self.conversation_log)
            self._append_log("floria", ai_reply)

        # ラウンド更新
        self.state["round"] += 1

        return ai_reply
