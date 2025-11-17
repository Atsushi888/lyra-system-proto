# actors/actor.py

from __future__ import annotations
from typing import List, Dict, Any

from personas.persona_floria_ja import Persona
from llm.llm_router import LLMRouter
from actors.answer_talker import AnswerTalker


class Actor:
    """
    Lyra System 上の「登場人物」。
    - 会話ログからプレイヤー発言を取得
    - Persona でプロンプトを組み立て
    - AnswerTalker に渡して最終返却を受け取る
    """

    def __init__(
        self,
        name: str,
        persona: Persona,
        router: LLMRouter | None = None,
    ) -> None:
        self.name = name
        self.persona = persona
        self.router = router or LLMRouter()

        # AnswerTalker を保持（パイプライン全体の司令塔）
        self.answer_talker = AnswerTalker()

    def speak(self, conversation_log: List[Dict[str, str]]) -> str:
        """
        会話ログをもとに Actor が一発話す。
        Actor は AI パイプラインを直接呼ばず、
        AnswerTalker に user_text を渡すだけにする。
        """

        # 最新のプレイヤー発言を取得
        user_text = ""
        for entry in reversed(conversation_log):
            if entry.get("role") == "player":
                user_text = entry.get("content", "")
                break

        # Persona に渡すプロンプトを生成
        messages = self.persona.build_messages(user_text)

        # GPT-4o へ一次問い合わせ（従来どおり）
        result = self.router.call_gpt4o(messages)

        # reply_text 抽出
        reply_text = None
        if isinstance(result, tuple):
            reply_text = result[0]
        else:
            reply_text = result

        # ★ AnswerTalker に処理を委ねる（最終返却を得る）
        final_text = self.answer_talker.speak(
            reply_text=reply_text,
            raw_result=result,
            user_text=user_text,               # ★ Add
        )

        return reply_text
