# actors/actor.py

from __future__ import annotations

from typing import List, Dict, Any

from personas.persona_floria_ja import Persona
from llm.llm_router import LLMRouter
from actors.answer_talker import AnswerTalker


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

        # ★ AnswerTalker を保持（将来ここから Judge / Composer を駆動）
        self.answer_talker = AnswerTalker()

    def speak(self, conversation_log: List[Dict[str, str]]) -> str:
        """
        会話ログをもとに、この Actor が一発話す。

        conversation_log:
            [{"role": "player" / "floria" / "system", "content": "..."}, ...]
        """
        # ひとまず、プレイヤーの最後の発言だけを見る簡易版で OK。
        user_text = ""
        for entry in reversed(conversation_log):
            if entry.get("role") == "player":
                user_text = entry.get("content", "")
                break

        # Persona から messages を組み立て
        messages = self.persona.build_messages(user_text)

        # LLMRouter 経由で GPT-4o を呼ぶ（現状のメイン処理）
        result = self.router.call_gpt4o(messages)

        # 戻り値の形に応じて柔軟に受け取る
        reply_text = None
        _usage = None
        _meta = {}

        if isinstance(result, tuple):
            if len(result) == 3:
                reply_text, _usage, _meta = result
            elif len(result) == 2:
                reply_text, _usage = result
                _meta = {}
            elif len(result) == 1:
                reply_text = result[0]
            else:
                # それ以上は想定外だけど、とりあえず先頭だけ採用
                reply_text = result[0]
        else:
            # ただの文字列やオブジェクトが返ってきた場合
            reply_text = result

        # ★ ここから先は AnswerTalker に委譲
        #   いまは AnswerTalker.speak() は「何もしないで reply_text を返す」だけ。
        final_text = self.answer_talker.speak(reply_text=reply_text, raw_result=result)

        return final_text
