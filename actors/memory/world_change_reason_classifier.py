# actors/memory/world_change_reason_classifier.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from llm.llm_manager import LLMManager


class WorldChangeReasonClassifier:
    """
    世界変化が起きたが、明確な reasons（発言トリガ）が抽出できなかった場合に、
    その原因を 2 択で分類する補助分類器。

    Returns:
        - interpersonal_complexity
        - external_event
    """

    def __init__(
        self,
        *,
        persona_id: str = "default",
        preferred_model: str = "gpt52",
        llm: Optional[LLMManager] = None,  # ★外部から注入できるように（MemoryAI互換）
    ) -> None:
        self.persona_id = persona_id
        self.preferred_model = preferred_model
        self._llm: LLMManager = llm or LLMManager.get_or_create(persona_id=persona_id)

        # 長い会話で暴発しないためのクリップ設定
        self.MAX_LINES = 28
        self.MAX_CHARS_PER_LINE = 600

    def classify(
        self,
        messages: List[Dict[str, Any]],
        final_reply: str,
    ) -> str:
        model = self._pick_model()

        system_prompt = (
            "You are a strict classifier.\n"
            "Return ONLY one of the following strings:\n"
            "- interpersonal_complexity\n"
            "- external_event\n"
        )

        user_prompt = self._build_prompt(messages, final_reply)

        text, _ = self._llm.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=16,
        )

        s = (text or "").strip().lower()

        if "interpersonal" in s:
            return "interpersonal_complexity"
        if "external" in s:
            return "external_event"

        # 判定不能時の安全側フォールバック
        return "external_event"

    # -------------------------------------------------
    # internals
    # -------------------------------------------------
    def _pick_model(self) -> str:
        props = self._llm.get_available_models() or {}

        candidates = [
            self.preferred_model,
            "gpt52",
            "gpt51",
            "gpt4o",
            "gemini",
            "grok",
        ]
        for m in candidates:
            if m in props:
                return m
        return self.preferred_model

    @staticmethod
    def _clip(s: str, n: int) -> str:
        if not s:
            return ""
        s = str(s)
        return s if len(s) <= n else (s[: n - 1] + "…")

    def _build_prompt(
        self,
        messages: List[Dict[str, Any]],
        final_reply: str,
    ) -> str:
        """
        - 直近側を優先して最大 MAX_LINES まで拾う
        - 1行の長さを MAX_CHARS_PER_LINE にクリップ
        - 役割は user / assistant のみ対象
        """
        picked: List[str] = []

        for m in reversed(messages or []):
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            content = (m.get("content") or "").strip()
            if not content:
                continue

            content = self._clip(content, self.MAX_CHARS_PER_LINE)
            picked.append(f"<{str(role).upper()}> {content}")

            if len(picked) >= self.MAX_LINES:
                break

        picked.reverse()

        fr = self._clip((final_reply or "").strip(), self.MAX_CHARS_PER_LINE)
        if fr:
            picked.append(f"<ASSISTANT_FINAL> {fr}")

        return (
            "A world-level change has been detected, but no single utterance can be used "
            "as its explicit reason.\n\n"
            "Classify the primary cause as EXACTLY one of:\n"
            "- interpersonal_complexity (complex relationships, implicit subtext, multi-party conflict)\n"
            "- external_event (disaster, accident, environmental/social events)\n\n"
            "Conversation (most recent, clipped):\n"
            + "\n".join(picked)
        )
