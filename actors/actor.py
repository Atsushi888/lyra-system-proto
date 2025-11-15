# actors/actor.py

from __future__ import annotations
from typing import List, Dict, Any
import os

from openai import OpenAI


# ==== OpenAI クライアント設定 ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY が設定されていないため Actor を初期化できません。")

# メインモデル（なければ gpt-4o を使う）
OPENAI_MAIN_MODEL = os.getenv("OPENAI_MAIN_MODEL", "gpt-4o")

_client = OpenAI(api_key=OPENAI_API_KEY)


def _call_gpt4o(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 600,
) -> tuple[str, Dict[str, Any]]:
    """
    会談用のシンプルな GPT 呼び出しヘルパ。
    返り値: (text, meta)
    """
    resp = _client.chat.completions.create(
        model=OPENAI_MAIN_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    usage: Dict[str, Any] = {}
    try:
        # usage があればメタに入れておく（無くても動く）
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
    except Exception:
        pass

    meta = {
        "route": "gpt",
        "model_name": OPENAI_MAIN_MODEL,
        "usage": usage,
    }
    return text, meta


class Actor:
    """
    会談システム用のシンプルな「話し手」クラス。

    - name: 画面表示用の名前（例: "フローリア"）
    - persona: いずれ system プロンプト生成に使う前提で保持しておく
    """

    def __init__(self, name: str, persona: object | None = None) -> None:
        self.name = name
        self.persona = persona

    # --- LLM に渡す messages を組み立て ---
    def _build_messages(self, conversation_log: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        いまのところ「直近のプレイヤー発言」だけを user に渡す。
        前後文脈などは、後でちゃんと設計するときに拡張する。
        """
        last_user_text = ""
        for m in reversed(conversation_log):
            if m.get("role") == "player":
                last_user_text = m.get("content", "")
                break

        # HTML の <br> を改行に戻してから LLM に渡す
        last_user_text = last_user_text.replace("<br>", "\n")

        # ★ persona 未使用だけど、とりあえず直書き system プロンプト
        system_prompt = (
            "あなたは『フローリア』という名の、水と氷の精霊の乙女です。"
            "プレイヤーの恋人として、優しく、穏やかに、日本語で会話してください。\n"
            "一人称は「わたし」。\n"
            "会話はロマンチックで感情豊かに。ただし露骨な性描写は用いず、"
            "ソフトで甘い雰囲気を大事にしてください。"
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": last_user_text or "プレイヤーからの言葉に続けて、自然に会話してください。",
            },
        ]
        return messages

    # --- 会話を 1 ターン進める ---
    def speak(self, conversation_log: List[Dict[str, str]]) -> str:
        messages = self._build_messages(conversation_log)
        text, _meta = _call_gpt4o(
            messages=messages,
            temperature=0.7,
            max_tokens=600,
        )
        return text
