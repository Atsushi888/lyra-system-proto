# council/council_manager.py

from __future__ import annotations
from typing import List, Dict, Any, Optional

from actors.actor import Actor
from personas.persona_floria_ja import Persona


class CouncilManager:
    """
    会談システムのロジック中核（β）。

    - conversation_log: [{ "role": "player" | "floria" | "system", "content": "<br>付き本文" }, ...]
    - state:
        round        : 直近の発言番号（conversation_log の長さ）
        speaker      : 次に話すべき話者（今は常に "player" 固定運用）
        mode         : "ongoing" / "ended" など（今は "ongoing" のみ使用）
        participants : 参加者リスト
        last_speaker : 直近に発言した話者
    """

    def __init__(self) -> None:
        # 会話ログ
        self.conversation_log: List[Dict[str, str]] = []

        # 参加 Actor（とりあえずフローリアだけ）
        self.actors: Dict[str, Actor] = {
            "floria": Actor("フローリア", Persona()),
        }

        # 会談状態
        self.state: Dict[str, Any] = {
            "round": 0,                      # ★ 発言数と一致させる
            "speaker": "player",
            "mode": "ongoing",
            "participants": ["player", "floria"],
            "last_speaker": None,
        }

    # ===== 基本操作 =====

    def reset(self) -> None:
        """会談を最初からやり直す。"""
        self.conversation_log.clear()
        self.state.update(
            {
                "round": 0,
                "speaker": "player",
                "mode": "ongoing",
                "participants": ["player", "floria"],
                "last_speaker": None,
            }
        )

    def _append_log(self, role: str, content: str) -> None:
        """
        ログに 1 発言追加し、round / last_speaker を更新する。
        改行は <br> に変換しておく（表示側で markdown + unsafe_allow_html を使う前提）。
        """
        safe_text = content.replace("\n", "<br>")
        self.conversation_log.append(
            {
                "role": role,
                "content": safe_text,
            }
        )
        # 発言番号は conversation_log の長さと一致させる
        self.state["last_speaker"] = role
        self.state["round"] = len(self.conversation_log)

    # ===== メイン処理 =====

    def proceed(self, user_text: str) -> Optional[str]:
        """
        プレイヤーの発言を受け取り、フローリアの返答までをまとめて処理する。

        戻り値: フローリアの返答（Actor が存在しなければ None）
        """
        user_text = user_text or ""
        user_text = user_text.strip()
        if not user_text:
            return None

        # 1) プレイヤー発言をログに追加
        self._append_log("player", user_text)

        # 2) フローリア Actor による応答生成
        actor = self.actors.get("floria")
        ai_reply: Optional[str] = None
        if actor is not None:
            ai_reply = actor.speak(self.conversation_log) or ""
            self._append_log("floria", ai_reply)

        # 3) 次のターンの話者はプレイヤーに戻しておく
        self.state["speaker"] = "player"

        return ai_reply
