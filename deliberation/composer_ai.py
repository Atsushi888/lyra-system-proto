# deliberation/composer_ai.py

from __future__ import annotations

from typing import Any, Dict, Optional

from llm_router import call_with_fallback


class ComposerAI:
    """
    複数モデルの応答と Judge の結果をもとに、
    「最終的な1本のフローリア返答候補」を決めるクラス。

    v1:
      - Judge の winner をそのまま採用
      - winner が取れない場合は gpt4o を優先
      - それもなければ先頭のモデルを採用

    ここでは Lyra 本体の返答はまだ差し替えず、
    あくまで「裏画面でのベスト候補表示」に使う前提。
    """

    def __init__(
        self,
        mode: str = "winner_only",  # 将来 "compose" に拡張するかもしれない
        temperature: float = 0.7,
        max_tokens: int = 800,
    ) -> None:
        self.mode = mode
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    # ---- 公開インターフェース ----
    def decide_final_reply(
        self,
        user_prompt: str,
        models: Dict[str, Any],
        judge: Optional[Dict[str, Any]],
        base_reply: str,
    ) -> Dict[str, Any]:
        """
        最終返答候補とメタ情報を返す。

        戻り値例:
        {
            "final_reply": "...",
            "chosen_model": "gpt4o",
            "mode": "winner_only",
        }
        """

        if not isinstance(models, dict) or not models:
            # models がない場合は、今まで通りの base_reply を返す
            return {
                "final_reply": base_reply,
                "chosen_model": "base",
                "mode": "fallback",
            }

        # ---- v1: winner_only モード ----
        if self.mode == "winner_only":
            chosen_key = self._choose_by_winner(judge, models)
            chosen_reply = str(models[chosen_key].get("reply") or base_reply)

            return {
                "final_reply": chosen_reply,
                "chosen_model": chosen_key,
                "mode": "winner_only",
            }

        # ---- 将来用: compose モード（今は使わない） ----
        if self.mode == "compose":
            return self._compose_with_llm(
                user_prompt=user_prompt,
                models=models,
                judge=judge,
                base_reply=base_reply,
            )

        # 不明なモード → フォールバック
        return {
            "final_reply": base_reply,
            "chosen_model": "base",
            "mode": "unknown_mode",
        }

    # ---- 内部ヘルパ ----
    def _choose_by_winner(
        self,
        judge: Optional[Dict[str, Any]],
        models: Dict[str, Any],
    ) -> str:
        """
        Judge の winner を見て、採用するモデルキーを決める。
        なければ gpt4o → 先頭の順。
        """
        # 1) Judge の winner
        if isinstance(judge, dict):
            winner = judge.get("winner")
            if isinstance(winner, str) and winner in models:
                return winner

        # 2) gpt4o があれば優先
        if "gpt4o" in models:
            return "gpt4o"

        # 3) それもなければ先頭
        keys = list(models.keys())
        return keys[0]

    def _compose_with_llm(
        self,
        user_prompt: str,
        models: Dict[str, Any],
        judge: Optional[Dict[str, Any]],
        base_reply: str,
    ) -> Dict[str, Any]:
        """
        将来用: 複数候補を LLM に渡して、
        「よりよい最終返答」を合成してもらうモード。
        v1 では呼ばない。
        """
        base_text = base_reply

        others_text = ""
        for key, info in models.items():
            reply = str(info.get("reply") or "")
            others_text += f"\n\n【候補 {key}】\n{reply}"

        system = (
            "あなたは物語エンジン Lyra の編集AIです。\n"
            "ユーザーのプロンプトと、複数の候補応答を読み比べ、\n"
            "フローリアとして最も自然で魅力的な最終返答を1つだけ作成してください。\n"
            "文体と人格はフローリアのキャラクターを保ちます。"
        )

        user = (
            "【ユーザープロンプト】\n"
            f"{user_prompt}\n\n"
            "【ベース候補】\n"
            f"{base_text}\n"
            f"{others_text}\n\n"
            "これらを踏まえ、より良い1つの最終返答を日本語で出力してください。"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        text, meta = call_with_fallback(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return {
            "final_reply": text,
            "chosen_model": "composed",
            "mode": "compose",
            "compose_meta": meta,
        }
