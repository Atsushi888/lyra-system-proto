# council/council_manager.py

from __future__ import annotations
from typing import List, Dict, Any

from actors.actor import Actor
from personas.persona_floria_ja import Persona


class CouncilManager:
    """
    会談システムのロジック側（β）。
    - conversation_log: 会話の生ログ（プレイヤー/フローリア両方）
    - round は「発言の総数」として len(conversation_log) から毎回計算する
    """

    def __init__(self) -> None:
        # 会話ログ：List[{"role": "...", "content": "..."}]
        self.conversation_log: List[Dict[str, str]] = []

        # いまはフローリア AI だけ
        self.actors: Dict[str, Actor] = {
            "floria": Actor("フローリア", Persona())
        }

        # 状態（round は持たず、都度計算）
        self.state: Dict[str, Any] = {
            "mode": "ongoing",
            "participants": ["player", "floria"],
            "last_speaker": None,
        }

    # ===== 内部ヘルパ =====
    def _append_log(self, role: str, content: str) -> None:
        """ログに 1 発言を追加。改行は <br> に変換して保存。"""
        safe = (content or "").replace("\n", "<br>")
        self.conversation_log.append({"role": role, "content": safe})
        self.state["last_speaker"] = role

    # ===== 外向け API =====
    def reset(self) -> None:
        """会談を最初からやり直す。"""
        self.conversation_log.clear()
        self.state["mode"] = "ongoing"
        self.state["last_speaker"] = None

    def get_log(self) -> List[Dict[str, str]]:
        """会談ログのコピーを返す（表示用）。"""
        return list(self.conversation_log)

    def get_status(self) -> Dict[str, Any]:
        """
        サイドバー表示用のステータス。
        round は「これからプレイヤーが行う発言の番号」として計算する。
        """
        # すでに終わった発言数 + 1 = 次の自分の発言番号
        round_ = len(self.conversation_log) + 1

        return {
            "round": round_,
            "speaker": "player",  # いまは常にプレイヤーのターン開始とみなす
            "mode": self.state.get("mode", "ongoing"),
            "participants": self.state.get("participants", ["player", "floria"]),
            "last_speaker": self.state.get("last_speaker"),
        }

    def proceed(self, user_text: str) -> str:
        """
        プレイヤーの発言を受け取り、
        - ログに追加
        - フローリアに conversation_log 丸ごと渡して返事を生成
        - 返事もログに追加
        を行う。
        """
        # プレイヤー発言
        self._append_log("player", user_text)

        reply = ""
        actor = self.actors.get("floria")
        if actor is not None:
            reply = actor.speak(self.conversation_log)
            self._append_log("floria", reply)

        return reply
