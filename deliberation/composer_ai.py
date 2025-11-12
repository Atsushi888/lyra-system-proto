# deliberation/composer_ai.py

from __future__ import annotations

from typing import Any, Dict, Optional


class ComposerAI:
    """
    複数モデルの応答と Judge の結果を見て、
    「最終候補テキスト」を決めるための小さなヘルパ。

    v1 ではまだ Lyra 本編の返答は差し替えるかどうかを
    呼び出し側で決める。ここでは final_reply を返すだけ。
    """

    def __init__(self, mode: str = "winner_only") -> None:
        # mode:
        #   - "winner_only": Judge の winner モデルの返答をそのまま採用
        self.mode = mode

    def decide_final_reply(
        self,
        user_prompt: str,
        models: Dict[str, Any],
        judge: Optional[Dict[str, Any]],
        base_reply: str,
    ) -> Dict[str, Any]:
        """
        最終候補テキストを決めて dict で返す。

        戻り値例:
        {
            "mode": "winner_only",
            "chosen_model": "hermes",
            "final_reply": "...",
        }
        """
        final_reply: str = base_reply
        chosen_model: str = self._default_chosen_model(models)

        if self.mode == "winner_only":
            winner = None
            if isinstance(judge, dict):
                w = judge.get("winner")
                if isinstance(w, str):
                    winner = w

            if winner and winner in models:
                model_info = models[winner] or {}
                reply = (
                    model_info.get("reply")
                    or model_info.get("text")
                    or ""
                )
                reply = str(reply)

                if reply.strip():
                    final_reply = reply
                    chosen_model = winner

        return {
            "mode": self.mode,
            "chosen_model": chosen_model,
            "final_reply": final_reply,
        }

    def _default_chosen_model(self, models: Dict[str, Any]) -> str:
        """
        何も決められなかった場合の「とりあえずのモデル名」。
        gpt4o がいれば gpt4o、なければ先頭キー。
        """
        if "gpt4o" in models:
            return "gpt4o"
        if models:
            return list(models.keys())[0]
        return "unknown"
