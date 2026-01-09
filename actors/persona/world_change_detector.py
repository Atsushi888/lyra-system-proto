" actors/persona/world_change_detector.py

from __future__ import annotations

from typing import Any, Dict, List, Optional


class WorldChangeDetector:
    """
    Persona 非依存の世界変化検出器（v0.1 ルールベース）。

    - 世界観・関係性・存在状態が「不可逆に変わった」かどうかだけを見る
    - importance=5 に相当
    """

    def __init__(self, persona_raw: Optional[Dict[str, Any]] = None) -> None:
        self.persona_raw = persona_raw or {}

        self.keywords = set(
            self.persona_raw.get("world_change_keywords", [])
            or [
                "死",
                "別れ",
                "永遠",
                "結婚",
                "崩壊",
                "取り返しがつかない",
                "世界が変わった",
            ]
        )

    def detect(
        self,
        messages: List[Dict[str, str]],
        final_reply: str,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
              "is_world_change": bool,
              "reasons": Optional[List[str]],
            }
        """
        reasons: List[str] = []

        for m in messages or []:
            if m.get("role") != "user":
                continue
            text = (m.get("content") or "").strip()
            if not text:
                continue
            if any(k in text for k in self.keywords):
                reasons.append(text)

        if not reasons and final_reply:
            if any(k in final_reply for k in self.keywords):
                reasons.append(final_reply.strip())

        if reasons:
            return {
                "is_world_change": True,
                "reasons": reasons,
            }

        return {
            "is_world_change": False,
            "reasons": None,
        }
