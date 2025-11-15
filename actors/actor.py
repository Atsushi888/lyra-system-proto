# actors/actor.py

from __future__ import annotations

from typing import List, Dict, Any

from personas.persona_floria_ja import Persona
from llm.llm_router import LLMRouter


class Actor:
    """
    Lyra System 上の「登場人物」1人を表現するクラス。

    - name: 画面に出す名前（例: "フローリア"）
    - persona: そのキャラ用のペルソナ定義
    - router: そのキャラが使う LLMRouter
              ※ 省略時は内部で自前の LLMRouter を作る
    """

    def __init__(
        self,
        name: str,
        persona: Persona,
        router: LLMRouter | None = None,
    ) -> None:
        self.name = name
        self.persona = persona
        # router 引数が省略された場合は、自前でインスタンスを作る
        self.router = router or LLMRouter()

    def speak(self, conversation_log: List[Dict[str, str]]) -> str:
        """
        会話ログをもとに、この Actor が一発話す。

        conversation_log:
            [{"role": "player" / "floria" / "system", "content": "..."}, ...]
        """
        # 必要に応じて、ここで conversation_log を
        # OpenAI 用 messages に変換して router に渡す。
        # ひとまず、プレイヤーの最後の発言だけを見る簡易版で OK。
        user_text = ""
        for entry in reversed(conversation_log):
            if entry.get("role") == "player":
                user_text = entry.get("content", "")
                break

        messages = self.persona.build_messages(user_text)
        reply_text, _usage, _meta = self.router.call_gpt4o(messages)

        return reply_text
