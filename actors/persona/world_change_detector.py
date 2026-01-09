# actors/persona/world_change_detector.py
from __future__ import annotations

from typing import Any, Dict, List, Optional


class WorldChangeDetector:
    """
    Persona 非依存の世界変化検出器（v0.1 / ルールベース）。

    - 世界観・関係性・存在状態が「不可逆に変わった」可能性だけを見る
    - importance=5 の候補抽出用
    """

    def __init__(self, persona_raw: Optional[Dict[str, Any]] = None) -> None:
        self.persona_raw = persona_raw or {}

        default_keywords = [
            "死",
            "別れ",
            "永遠",
            "結婚",
            "崩壊",
            "取り返しがつかない",
            "世界が変わった",
        ]

        # Persona 側で追加定義できる（空/未定義ならデフォルトのみ）
        extra = self.persona_raw.get("world_change_keywords", [])
        extra_list: List[str] = [str(x) for x in extra] if isinstance(extra, list) else []

        # 「デフォルト + 追加」を統合（persona_raw が入ってもデフォルトを殺さない）
        self.keywords = set(default_keywords) | set(extra_list)

    def detect(
        self,
        messages: List[Dict[str, Any]],
        final_reply: str,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
              "is_world_change": bool,
              "reasons": List[str],   # 常に list（空の可能性あり）
            }
        """
        reasons: List[str] = []

        # user 発話からヒットを拾う
        for m in messages or []:
            if not isinstance(m, dict):
                continue
            if m.get("role") != "user":
                continue

            text = str(m.get("content") or "").strip()
            if not text:
                continue

            if any(k in text for k in self.keywords):
                reasons.append(text)

        # user で拾えなければ最終返答も見る（保険）
        fr = (final_reply or "").strip()
        if not reasons and fr:
            if any(k in fr for k in self.keywords):
                reasons.append(fr)

        return {
            "is_world_change": bool(reasons),
            "reasons": reasons,
        }
