# actors/memory/world_change_reason_classifier.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from llm.llm_manager import LLMManager


class WorldChangeReasonClassifier:
    """
    世界変化が起きたが、明確な reasons（発言トリガ）が抽出できなかった場合に、
    その原因を 2 択で分類する補助分類器。

    Returns:
        - interpersonal_complexity
        - external_event

    運用上の安全策:
    - 入力ログの長さをクリップして LLM の暴走/失敗を避ける
    - 返答の揺れを吸収（"external event", "EXTERNAL_EVENT.", JSON など）
    - モデル選択は available_models を参照し、なければ順次フォールバック
    """

    # Conversation の最大行数（プロンプト肥大化対策）
    MAX_LINES: int = 40
    # 1行あたりの最大文字数（肥大化対策）
    MAX_CHARS_PER_LINE: int = 240
    # final_reply の最大文字数（肥大化対策）
    MAX_FINAL_CHARS: int = 500

    def __init__(
        self,
        *,
        persona_id: str = "default",
        preferred_model: str = "gpt52",
        llm: Optional[LLMManager] = None,
    ) -> None:
        self.persona_id = persona_id
        self.preferred_model = preferred_model
        self._llm = llm or LLMManager.get_or_create(persona_id=persona_id)

    def classify(
        self,
        messages: List[Dict[str, Any]],
        final_reply: str,
    ) -> str:
        model = self._pick_model()

        system_prompt = (
            "You are a strict classifier.\n"
            "Return ONLY one of the following tokens:\n"
            "interpersonal_complexity\n"
            "external_event\n"
            "No extra text."
        )

        user_prompt = self._build_prompt(messages, final_reply)

        try:
            text, _usage = self._llm.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=16,
            )
        except Exception:
            # 呼び出し自体が落ちたら安全側（外因）に倒す
            return "external_event"

        return self._parse_label(text)

    # ---------------- internal ----------------

    def _pick_model(self) -> str:
        props = self._llm.get_available_models() or {}

        # 念のため normalize（キーの大小揺れや空白を吸収）
        available = {str(k).strip().lower() for k in props.keys()}

        candidates = [
            self.preferred_model,
            "gpt52",
            "gpt51",
            "gemini",
            "grok",
        ]

        for m in candidates:
            key = (m or "").strip().lower()
            if key and (key in available):
                return m

        # available_models が空/未初期化でも preferred を返す（呼び出し側で落ちる可能性はあるが、ここは責務外）
        return self.preferred_model

    @staticmethod
    def _parse_label(text: Any) -> str:
        s = "" if text is None else str(text)
        s = s.strip().lower()

        # JSON っぽく返ってくる事故も吸収
        # 例: {"label":"external_event"} / label: external_event
        if "interpersonal_complexity" in s or "interpersonal" in s:
            return "interpersonal_complexity"
        if "external_event" in s or "external event" in s or "external" in s:
            return "external_event"

        # 判定不能時の安全側フォールバック
        return "external_event"

    @classmethod
    def _clip(cls, s: str, max_chars: int) -> str:
        t = (s or "").strip()
        if len(t) <= max_chars:
            return t
        return t[: max_chars - 1] + "…"

    def _build_prompt(
        self,
        messages: Sequence[Dict[str, Any]],
        final_reply: str,
    ) -> str:
        lines: List[str] = []

        # 末尾側（直近）から拾って、最後に時系列順へ戻す
        picked: List[str] = []
        for m in reversed(list(messages or [])):
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "").strip().lower()
            if role not in ("user", "assistant"):
                continue

            content = self._clip(str(m.get("content") or ""), self.MAX_CHARS_PER_LINE)
            if not content:
                continue

            picked.append(f"<{role.upper()}> {content}")

            if len(picked) >= self.MAX_LINES:
                break

        picked.reverse()
        lines.extend(picked)

        fr = self._clip(str(final_reply or ""), self.MAX_FINAL_CHARS)
        if fr:
            lines.append(f"<ASSISTANT_FINAL> {fr}")

        return (
            "A world-level change has been detected, but no single utterance can be used "
            "as its explicit reason.\n\n"
            "Classify the primary cause as EXACTLY ONE token:\n"
            "- interpersonal_complexity  (complex interpersonal dynamics, relationship shifts, "
            "implicit subtext, multi-party context)\n"
            "- external_event            (disasters, unavoidable circumstances, outside forces, "
            "environmental/social events)\n\n"
            "Conversation (most recent, clipped):\n"
            + "\n".join(lines)
        )
