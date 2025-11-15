# actors/actor.py

from __future__ import annotations
from typing import List, Dict, Any
import os

from openai import OpenAI

from personas.persona_floria_ja import Persona  # 実際の型名はそのままでOK
from llm.llm_router import LLMRouter



# ==== OpenAI クライアント設定 ====

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY が設定されていないため Actor を初期化できません。")

# メインモデル（未設定なら gpt-4o）
OPENAI_MAIN_MODEL = os.getenv("OPENAI_MAIN_MODEL", "gpt-4o")

_client = OpenAI(api_key=OPENAI_API_KEY)


def _call_gpt4o(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 600,
) -> tuple[str, Dict[str, Any]]:
    """
    会談用のシンプルな GPT 呼び出し。
    戻り値: (text, meta)
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
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
    except Exception:
        pass

    meta: Dict[str, Any] = {
        "route": "gpt",
        "model_name": OPENAI_MAIN_MODEL,
        "usage": usage,
    }
    return text, meta


class Actor:
    """
    会談システム用「話し手」クラス。

    - name: 画面表示用の名前（例: "フローリア"）
    - persona: いまは未使用だが、将来 system プロンプト生成等に使うため保持
    """
    def __init__(self, name: str, persona: Persona) -> None:
        self.name = name
        self.persona = persona
        # 各 Actor は自分専用の LLMRouter を持つ
        self.router = LLMRouter()

    # ---- LLM に投げるメッセージを組み立て ----
    def _build_messages(self, conversation_log: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        いまは「直近のプレイヤー発言」だけを user に渡す簡易版。
        前後文脈はあとで拡張する。
        """
        last_user_text = ""
        for m in reversed(conversation_log):
            if m.get("role") == "player":
                last_user_text = m.get("content", "")
                break

        # LLM には <br> ではなく改行で渡す
        last_user_text = last_user_text.replace("<br>", "\n")

        # ひとまず persona はまだ使わず、固定の system プロンプト
        system_prompt = (
            "あなたは『フローリア』という名の、水と氷の精霊の乙女です。"
            "プレイヤーの恋人として、優しく穏やかに日本語で会話してください。\n"
            "一人称は「わたし」。\n"
            "会話はロマンチックで感情豊かに。ただし露骨な性描写は避け、"
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

    # ---- 1ターン発言 ----
    def speak(self, conversation_log: List[Dict[str, str]]) -> str:
        messages = self._build_messages(conversation_log)
        text, _meta = _call_gpt4o(
            messages=messages,
            temperature=0.7,
            max_tokens=600,
        )
        return text
