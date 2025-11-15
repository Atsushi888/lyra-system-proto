# council/council_manager.py

from __future__ import annotations
from typing import List, Dict, Any

from actors.actor import Actor
from personas.persona_floria_ja import Persona


class CouncilManager:
    """
    会談システムのロジック側（β）。
    - conversation_log: 画面に出すログ
    - state: ラウンド数や参加者などのメタ情報
    """

    def __init__(self) -> None:
        # 会話ログ：List[{role:"player"|"floria", content:"..."}]
        self.conversation_log: List[Dict[str, str]] = []

        # ひとまずフローリア1人だけ
        self.actors: Dict[str, Actor] = {
            "floria": Actor("フローリア", Persona())
        }

        self.state: Dict[str, Any] = {
            "round": 1,
            "speaker": "player",
            "mode": "ongoing",
            "participants": ["player", "floria"],
            "last_speaker": None,
        }

    # ---- 内部ヘルパ ----
    def _append_log(self, role: str, content: str) -> None:
        # HTML 表示用に改行を <br> に変換
        safe_text = content.replace("\n", "<br>")
        self.conversation_log.append({"role": role, "content": safe_text})
        # ラウンド番号は「発言の番号」と一致させる
        self.state["round"] = len(self.conversation_log)
        self.state["last_speaker"] = role
        self.state["speaker"] = "player" if role != "player" else "floria"

    # ---- 公開 API ----
    def reset(self) -> None:
        self.conversation_log.clear()
        self.state.update(
            {
                "round": 1,
                "speaker": "player",
                "mode": "ongoing",
                "participants": ["player", "floria"],
                "last_speaker": None,
            }
        )

    def proceed(self, user_text: str) -> str:
        """
        プレイヤー発言 → フローリア応答 までをまとめて処理。
        戻り値はフローリアの発言テキスト。
        """
        user_text = (user_text or "").strip()
        if not user_text:
            return ""

        # プレイヤー発言をログへ
        self._append_log("player", user_text)

        # フローリアの番
        actor = self.actors.get("floria")
        ai_reply = ""
        if actor is not None:
            ai_reply = actor.speak(self.conversation_log)
            self._append_log("floria", ai_reply)

        return ai_reply

    # ---- View 向けのゲッター ----
    def get_log(self) -> List[Dict[str, str]]:
        return list(self.conversation_log)

    def get_state(self) -> Dict[str, Any]:
        return dict(self.state)
