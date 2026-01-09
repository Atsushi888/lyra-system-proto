# actors/persona/world_change_detector.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple


class WorldChangeDetector:
    """
    Persona 非依存の世界変化検出器（v0.2 / ルールベース）。

    - 世界観・関係性・存在状態が「不可逆に変わった」可能性だけを見る
    - importance=5 の候補抽出用
    - 戻り値の互換性維持:
        {
          "is_world_change": bool,
          "reasons": List[str],   # 常に list
        }
      追加情報（あっても壊れない）:
        - matched_keywords: List[str]
    """

    # reasons に入れる1発言の最大長（事故防止）
    MAX_REASON_CHARS = 400
    # reasons の最大件数（事故防止）
    MAX_REASONS = 8

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
        extra_list: List[str] = [str(x).strip() for x in extra] if isinstance(extra, list) else []
        extra_list = [x for x in extra_list if x]

        # 「デフォルト + 追加」を統合（persona_raw が入ってもデフォルトを殺さない）
        self.keywords: Set[str] = set(default_keywords) | set(extra_list)

    @staticmethod
    def _normalize_text(text: str) -> str:
        # 今は最低限：stripのみ（必要ならここで全角スペース等の正規化を増やせる）
        return (text or "").strip()

    def _hit(self, text: str) -> Tuple[bool, List[str]]:
        """
        text が keywords にヒットするか、ヒットしたキーワード一覧も返す。
        """
        hits: List[str] = []
        if not text:
            return False, hits
        for k in self.keywords:
            if k and (k in text):
                hits.append(k)
        return bool(hits), hits

    @staticmethod
    def _clip_reason(text: str, max_chars: int) -> str:
        s = text.strip()
        if len(s) <= max_chars:
            return s
        return s[: max_chars - 1] + "…"

    def detect(
        self,
        messages: List[Dict[str, Any]],
        final_reply: str,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
              "is_world_change": bool,
              "reasons": List[str],          # 常に list（空の可能性あり）
              "matched_keywords": List[str], # 追加情報（デバッグ用）
            }
        """
        reasons: List[str] = []
        matched: Set[str] = set()

        # user 発話からヒットを拾う（重複は潰す）
        seen_reasons: Set[str] = set()
        for m in messages or []:
            if not isinstance(m, dict):
                continue
            if m.get("role") != "user":
                continue

            raw = str(m.get("content") or "")
            text = self._normalize_text(raw)
            if not text:
                continue

            ok, hits = self._hit(text)
            if not ok:
                continue

            for h in hits:
                matched.add(h)

            clipped = self._clip_reason(text, self.MAX_REASON_CHARS)
            if clipped not in seen_reasons:
                reasons.append(clipped)
                seen_reasons.add(clipped)

            if len(reasons) >= self.MAX_REASONS:
                break

        # user で拾えなければ最終返答も見る（保険）
        if not reasons:
            fr = self._normalize_text(str(final_reply or ""))
            if fr:
                ok, hits = self._hit(fr)
                if ok:
                    for h in hits:
                        matched.add(h)
                    reasons.append(self._clip_reason(fr, self.MAX_REASON_CHARS))

        return {
            "is_world_change": bool(reasons),
            "reasons": reasons,  # 常に list
            "matched_keywords": sorted(matched),
        }
