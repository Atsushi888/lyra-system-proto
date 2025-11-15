# council/council_manager.py

from __future__ import annotations
from typing import List, Dict, Any

from actors.actor import Actor
from personas.persona_floria_ja import Persona, get_persona


class CouncilManager:
    """
    会談システムのロジック側（β）。
    Actor ベースで応答を生成する最小構成。
    """

    def __init__(self) -> None:
        # 会話ログ：List[{role: "...", content: "..."}]
        self.conversation_log: List[Dict[str, str]] = []

        # フローリアのペルソナを取得
        floria_persona: Persona = get_persona()

        # 一旦フローリアAIだけでよい
        self.actors: Dict[str, Actor] = {
            # Actor のコンストラクタが (name, persona) だという前提
            "floria": Actor("フローリア", floria_persona)
        }

        # 簡易ステート
        self.state: Dict[str, Any] = {
            "round": 1,
            "speaker": "player",   # 現在の話者（将来拡張用）
            "mode": "ongoing",     # "ongoing" / "ended" など
        }

    # ---- 会話ログ操作 ----
    def _append_log(self, role: str, content: str) -> None:
        """
        会話ログに1件追加する。ここで改行→<br>の整形もやっておく。
        """
        safe_text = content.replace("\n", "<br>")
        self.conversation_log.append(
            {
                "role": role,        # ★ 引数の role を素直に使う
                "content": safe_text,
            }
        )

    # ---- リセット ----
    def reset(self) -> None:
        """会談をリセットし、最初から開始可能な状態に戻す。"""
        self.conversation_log.clear()
        self.state = {
            "round": 1,
            "speaker": "player",
            "mode": "ongoing",
        }

    # ---- メイン処理 ----
    def proceed(self, user_text: str) -> str:
        """
        プレイヤー発言 → AI応答までをまとめて処理する。
        戻り値: フローリアの返答テキスト（<br> 整形前の生テキスト or 後でもOK）
        """

        # モードが進行中でない場合は何もしない（保険）
        if self.state.get("mode") != "ongoing":
            return ""

        # 1) プレイヤー発言をログに追加
        self._append_log("player", user_text)

        # 2) 返事を生成する AI（現状フローリアのみ）
        actor = self.actors.get("floria")
        ai_reply: str = ""

        if actor is not None:
            # Actor は conversation_log 全体を見て返事を作る前提
            ai_reply = actor.speak(self.conversation_log)

            # 3) フローリアの返答もログに追加
            self._append_log("floria", ai_reply)

        # 4) ラウンド更新
        self.state["round"] += 1

        # 5) 画面側で使いやすいように、AIの生返答を返しておく
        return ai_reply
