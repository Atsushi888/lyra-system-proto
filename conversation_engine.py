# conversation_engine.py — LLM 呼び出しを統括する会話エンジン層

from typing import Any, Dict, List, Tuple

from llm_router import call_with_fallback


class LLMConversation:
    """
    system プロンプト（フローリア人格など）と LLM 呼び出しをまとめた会話エンジン。
    GPT-4o の応答をメインとしつつ、
    同じ内容を Hermes(dummy) としても models に載せる。
    """

    def __init__(
        self,
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 800,
        style_hint: str = "",
    ) -> None:
        self.system_prompt = system_prompt
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.style_hint = style_hint.strip() if style_hint else ""

        # デフォルトのスタイル指針（persona に style_hint がない場合のみ使用）
        self.default_style_hint = (
            "あなたは上記の system プロンプトで定義されたキャラクターとして振る舞います。\n"
            "ユーザーは物語の本文（地の文と会話文）を日本語で入力します。\n"
            "直前のユーザーの発言や行動を読み、その続きを自然に描写してください。\n"
            "文体は自然で感情的に。見出し・記号・英語タグ（onstage:, onscreen: など）は使わず、"
            "純粋な日本語の物語文として出力してください。"
        )

    # ===== LLMに渡すmessageを構築 =====
    def build_messages(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        system_content = self.system_prompt
        effective_style_hint = self.style_hint or self.default_style_hint
        system_content += "\n\n" + effective_style_hint

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_content}
        ]

        last_user_content = None
        for m in reversed(history):
            if m.get("role") == "user":
                last_user_content = m.get("content", "")
                break

        if last_user_content:
            messages.append({"role": "user", "content": last_user_content})
        else:
            messages.append(
                {
                    "role": "user",
                    "content": "（ユーザーはまだ発言していません。"
                               "あなた＝フローリアとして、軽く自己紹介してください）",
                }
            )
        return messages

    # ===== GPT-4o に投げ、gpt4o & Hermes(dummy) を models に詰める =====
    def generate_reply(
        self,
        history: List[Dict[str, str]],
    ) -> Tuple[str, Dict[str, Any]]:
        messages = self.build_messages(history)

        # GPT-4o メイン
        text, meta = call_with_fallback(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        meta = dict(meta)

        meta["prompt_messages"] = messages
        meta["prompt_preview"] = "\n\n".join(
            f"[{m['role']}] {m['content'][:300]}"
            for m in messages
        )

        usage_main = meta.get("usage_main") or meta.get("usage") or {}

        # GPT-4o 本体
        gpt_entry = {
            "reply": text,
            "usage": usage_main,
            "route": meta.get("route", "gpt"),
            "model_name": meta.get("model_main", "gpt-4o"),
        }

        # Hermes は今はダミー：同じ内容を別モデル扱いにする
        hermes_entry = {
            "reply": text,
            "usage": usage_main,
            "route": "dummy-hermes",
            "model_name": "Hermes (dummy)",
        }

        meta["models"] = {
            "gpt4o": gpt_entry,
            "hermes": hermes_entry,
        }

        return text, meta
