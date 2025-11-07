# lyra_core.py
from typing import Any, Dict, List, Tuple
import streamlit as st

class LyraCore:
    """Lyra Engine の中核。1ターンの対話を統括する。"""

    def __init__(self, conversation_engine):
        self.conversation = conversation_engine

    def proceed_turn(self, user_text: str, state) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """ユーザー入力を受けて、LLMとの1ターン会話を処理する。"""
        # プレイヤーの発言を追加
        state["messages"].append({"role": "user", "content": user_text})

        try:
            # LLM呼び出し
            reply_text, meta = self.conversation.generate_reply(state["messages"])
        except Exception as e:
            reply_text = f"⚠️ 応答生成中にエラーが発生しました: {e}"
            meta = {"route": "error", "exception": str(e)}

        # 応答空白時フォールバック
        if not reply_text or not reply_text.strip():
            reply_text = "……うまく返答を生成できなかったみたい。もう一度試してくれる？"

        # フローリアの返答を追加
        state["messages"].append({"role": "assistant", "content": reply_text})

        # メタ情報を保存
        state["llm_meta"] = meta
        return state["messages"], meta
