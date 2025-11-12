# conversation_engine.py — LLM 呼び出しを統括する会話エンジン層
from typing import Any, Dict, List, Tuple

from llm_router import call_with_fallback, call_hermes
from deliberation.judge_ai import JudgeAI
from deliberation.composer_ai import ComposerAI


class LLMConversation:
    """
    system プロンプト（フローリア人格など）と LLM 呼び出しをまとめた会話エンジン。

    現状：
      - GPT-4o と Hermes の両モデルを実際に呼び出す
      - JudgeAI が勝者を選定
      - ComposerAI が勝者モデルの返答を「本編出力」に採用
      - MultiAIResponse / DebugPanel でも同一メタ情報を利用
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

        # 審議・合成AIの初期化
        self.judge_ai = JudgeAI()
        self.composer = ComposerAI(mode="winner_only")

        # デフォルトのスタイル指針（persona に style_hint がない場合のみ使用）
        self.default_style_hint = (
            "あなたは上記の system プロンプトで定義されたキャラクターとして振る舞います。\n"
            "ユーザーは物語の本文（地の文と会話文）を日本語で入力します。\n"
            "直前のユーザーの発言や行動を読み、その続きを自然に描写してください。\n"
            "文体は自然で感情的に。見出し・記号・英語タグ（onstage:, onscreen: など）は使わず、"
            "純粋な日本語の物語文として出力してください。"
        )

    # ===== LLM に渡す messages を構築 =====
    def build_messages(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        「system（人格＋文体指針）」＋「最新 user 発言」だけを LLM に渡す。
        """
        system_content = self.system_prompt
        effective_style_hint = self.style_hint or self.default_style_hint
        system_content += "\n\n" + effective_style_hint

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_content}
        ]

        # 最新の user メッセージのみ抽出
        last_user_content: str | None = None
        for m in reversed(history):
            if m.get("role") == "user":
                last_user_content = m.get("content", "")
                break

        if last_user_content:
            messages.append({"role": "user", "content": last_user_content})
        else:
            messages.append({
                "role": "user",
                "content": (
                    "（ユーザーはまだ発言していません。"
                    "あなた＝フローリアとして、軽く自己紹介してください）"
                ),
            })

        return messages

    # ===== 複数モデル呼び出し＋審議統合 =====
    def generate_reply(
        self,
        history: List[Dict[str, str]],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        GPT-4o と Hermes 両方に投げ、Judge / Composer で最適応答を採用。
        """
        messages = self.build_messages(history)

        # --- GPT-4o 呼び出し ---
        text_gpt, meta_gpt = call_with_fallback(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # --- Hermes 呼び出し ---
        text_hermes, meta_hermes = call_hermes(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # --- メタ統合 ---
        meta: Dict[str, Any] = dict(meta_gpt)
        meta["prompt_messages"] = messages
        meta["prompt_preview"] = "\n\n".join(
            f"[{m['role']}] {m['content'][:300]}"
            for m in messages
        )

        usage_gpt = meta_gpt.get("usage_main") or {}
        usage_hermes = meta_hermes.get("usage_main") or {}

        # モデル一覧（審議対象）
        meta["models"] = {
            "gpt4o": {
                "reply": text_gpt,
                "usage": usage_gpt,
                "route": meta_gpt.get("route", "gpt"),
                "model_name": meta_gpt.get("model_main", "gpt-4o"),
            },
            "hermes": {
                "reply": text_hermes,
                "usage": usage_hermes,
                "route": meta_hermes.get("route", "openrouter"),
                "model_name": meta_hermes.get("model_main", "Hermes"),
            },
        }

        # --- JudgeAI による勝者判定 ---
        judge = self.judge_ai.run(meta)
        meta["judge"] = judge

        # --- ComposerAI による最終候補選出 ---
        user_prompt = ""
        if messages:
            user_prompt = messages[-1].get("content", "")

        composer_info = self.composer.decide_final_reply(
            user_prompt=user_prompt,
            models=meta["models"],
            judge=judge,
            base_reply=text_gpt,
        )
        meta["composer"] = composer_info

        # --- 最終出力テキストを勝者モデルに差し替え ---
        final_reply = composer_info.get("final_reply")
        if isinstance(final_reply, str) and final_reply.strip():
            text_final = final_reply
        else:
            text_final = text_gpt  # 念のためフォールバック

        # === 本編出力を「勝者モデルの返答」で返す ===
        return text_final, meta
