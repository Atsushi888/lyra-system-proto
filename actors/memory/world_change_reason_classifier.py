# actors/memory/world_change_reason_classifier.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from llm.llm_manager import LLMManager


ChatReturn = Union[str, Dict[str, Any], Tuple[Any, ...]]


class WorldChangeReasonClassifier:
    """
    世界変化が検出されたが、明確な reasons（発言トリガ）が抽出できなかった場合に、
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
        llm_manager: Optional[LLMManager] = None,
    ) -> None:
        self.persona_id = persona_id
        self.preferred_model = preferred_model
        self._llm = llm_manager or LLMManager.get_or_create(persona_id=persona_id)

    def classify(
        self,
        messages: List[Dict[str, str]],
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

        completion: ChatReturn = self._llm.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=16,
        )

        text = self._normalize_text(completion)
        s = (text or "").strip().lower()

        if "interpersonal" in s:
            return "interpersonal_complexity"
        if "external" in s:
            return "external_event"

        # 判定不能時の安全側フォールバック
        return "external_event"

    def _pick_model(self) -> str:
        props = self._llm.get_available_models() or {}

        candidates = [
            self.preferred_model,
            "gpt52",
            "gpt51",
            "gemini",
            "grok",
        ]
        for m in candidates:
            if m in props and (props.get(m) or {}).get("enabled", True):
                return m
        return self.preferred_model

    @staticmethod
    def _normalize_text(completion: ChatReturn) -> str:
        if isinstance(completion, dict):
            return str(completion.get("text") or completion.get("content") or "")
        if isinstance(completion, (tuple, list)):
            return "" if not completion else str(completion[0] or "")
        return "" if completion is None else str(completion)

    @staticmethod
    def _build_prompt(
        messages: List[Dict[str, str]],
        final_reply: str,
    ) -> str:
        lines: List[str] = []

        for m in messages or []:
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            content = (m.get("content") or "").strip()
            if content:
                lines.append(f"<{role.upper()}> {content}")

        fr = (final_reply or "").strip()
        if fr:
            lines.append(f"<ASSISTANT_FINAL> {fr}")

        return (
            "A world-level change has been detected, but no single utterance "
            "can be used as its explicit reason.\n\n"
            "Classify the primary cause as ONE of:\n"
            "1) interpersonal_complexity\n"
            "2) external_event\n\n"
            "Conversation:\n"
            + "\n".join(lines)
        )
