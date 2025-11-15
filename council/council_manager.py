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

    def __init__(self) -> None:
        # 会話ログ：List[{role:"", content:""}]
        self.conversation_log: List[Dict[str, str]] = []

        # ひとまずフローリア AI 一人だけ
        # ★ persona は渡しても渡さなくても動くが、ここでは渡さない形にしておく
        self.actors: Dict[str, Actor] = {
            "floria": Actor("フローリア")
            # Persona() を渡したい場合は: Actor("フローリア", Persona())
        }

        self.state: Dict[str, Any] = {
            "round": 1,
            "speaker": "player",   # 現在の話者（player / floria）
            "mode": "ongoing",     # idle / ongoing / ended
            "participants": ["player", "floria"],
            "last_speaker": None,
        }

    # ---- 会話ログ操作 ----
    def _append_log(self, role: str, content: str) -> None:
        """内部用：ログに1件追加（改行 → <br> に変換して保存）"""
        safe_text = content.replace("\n", "<br>")
        self.conversation_log.append(
            {
                "role": role,
                "content": safe_text,
            }
        )
        self.state["last_speaker"] = role

    # ---- リセット ----
    def reset(self) -> None:
        self.conversation_log.clear()
        self.state.update(
            {
                "round": 1,
                "speaker": "player",
                "mode": "ongoing",
                "last_speaker": None,
            }
        )

    # ---- メイン処理 ----
    def proceed(self, user_text: str) -> str:
        """
        プレイヤー発言 → フローリア応答までをまとめて処理。
        戻り値: フローリアの返答テキスト
        """
        if not user_text.strip():
            return ""

        # プレイヤー発言
        self._append_log("player", user_text)

        # 返事を生成する AI（いまはフローリア固定）
        actor = self.actors.get("floria")
        ai_reply = ""
        if actor:
            ai_reply = actor.speak(self.conversation_log)
            self._append_log("floria", ai_reply)

        # ラウンド更新（プレイヤーとフローリアで 1 ラウンド進めるイメージ）
        self.state["round"] += 1
        self.state["speaker"] = "player"

        return ai_reply

    # ---- ステータス取得（view から参照用）----
    def get_status(self) -> Dict[str, Any]:
        return dict(self.state)
