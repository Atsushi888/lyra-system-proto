# actors/memory_ai.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from llm.llm_manager import LLMManager
from actors.persona.world_change_detector import WorldChangeDetector
from actors.memory.world_change_reason_classifier import WorldChangeReasonClassifier


@dataclass
class MemoryRecord:
    id: str
    round_id: int
    importance: int
    summary: str
    tags: List[str]
    created_at: str
    source_user: str
    source_assistant: str

    # v1.1 追加
    world_change_reasons: Optional[List[str]] = None
    reason_unavailable: Optional[str] = None


class MemoryAI:
    """
    長期記憶管理（v1.1）

    - importance=5（世界変化）を最優先で処理
    - reasons が無い場合は外部分類器で reason_unavailable を付与

    互換性:
    - 旧呼び出し: MemoryAI(llm_manager, persona_id=..., model_name=...)
      を受けられるようにしてある
    """

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,  # ★互換：AnswerTalker が位置引数で渡す
        *,
        persona_id: str,
        persona_raw: Optional[Dict[str, Any]] = None,
        base_dir: str = "data/memory",
        max_store_items: int = 200,
        model_name: str = "gpt52",  # ★互換：AnswerTalker が渡す
    ) -> None:
        self.persona_id = persona_id
        self.persona_raw = persona_raw or {}
        self.max_store_items = max_store_items

        os.makedirs(base_dir, exist_ok=True)
        self.file_path = os.path.join(base_dir, f"{persona_id}.json")

        self.memories: List[MemoryRecord] = []

        # world change detector（ルールベース）
        self._detector = WorldChangeDetector(self.persona_raw)

        # reason classifier（LLM）
        # - llm_manager が渡されない場合は classifier 内で get_or_create する
        self._reason_classifier = WorldChangeReasonClassifier(
            persona_id=persona_id,
            preferred_model=model_name,
            llm=llm_manager,
        )

        self.load()

    # ---------------- persistence ----------------

    def load(self) -> None:
        if not os.path.exists(self.file_path):
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        if not isinstance(data, list):
            return

        self.memories = []
        for d in data:
            if not isinstance(d, dict):
                continue
            try:
                self.memories.append(
                    MemoryRecord(
                        id=str(d.get("id", "")),
                        round_id=int(d.get("round_id", 0)),
                        importance=int(d.get("importance", 1)),
                        summary=str(d.get("summary", "")),
                        tags=list(d.get("tags", [])),
                        created_at=str(d.get("created_at", "")),
                        source_user=str(d.get("source_user", "")),
                        source_assistant=str(d.get("source_assistant", "")),
                        world_change_reasons=d.get("world_change_reasons"),
                        reason_unavailable=d.get("reason_unavailable"),
                    )
                )
            except Exception:
                continue

    def save(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(m) for m in self.memories],
                f,
                ensure_ascii=False,
                indent=2,
            )

    # ---------------- main ----------------

    def update_from_turn(
        self,
        *,
        messages: List[Dict[str, Any]],
        final_reply: str,
        round_id: int,
    ) -> Dict[str, Any]:
        wc = self._detector.detect(messages, final_reply)

        user_text = self._extract_last_user(messages)
        base_text = (final_reply or "").strip() or (user_text or "").strip()

        if not base_text:
            return {"status": "skip", "added": 0}

        created_at = datetime.now(timezone.utc).isoformat()
        mem_id = f"{created_at}_{round_id}"

        importance = 5 if wc.get("is_world_change") else 4

        rec = MemoryRecord(
            id=mem_id,
            round_id=round_id,
            importance=importance,
            summary=base_text[:160] + ("…" if len(base_text) > 160 else ""),
            tags=["世界変化"] if importance == 5 else ["設定"],
            created_at=created_at,
            source_user=user_text,
            source_assistant=(final_reply or ""),
        )

        if importance == 5:
            reasons = wc.get("reasons") or []
            if isinstance(reasons, list) and reasons:
                rec.world_change_reasons = [str(x) for x in reasons][:10]
            else:
                # reasons が空＝「世界変化っぽいが明示トリガが拾えない」時のみ分類器を使う
                rec.reason_unavailable = self._reason_classifier.classify(
                    messages, final_reply
                )

        self.memories.append(rec)
        self._trim()
        self.save()

        return {
            "status": "ok",
            "added": 1,
            "importance": importance,
        }

    # ---------------- helpers ----------------

    @staticmethod
    def _extract_last_user(messages: List[Dict[str, Any]]) -> str:
        for m in reversed(messages or []):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content") or "")
        return ""

    def _trim(self) -> None:
        if len(self.memories) <= self.max_store_items:
            return
        # importance 昇順→created_at 昇順で並べ、後ろ（重要/新しい）を残す
        self.memories.sort(key=lambda m: (int(m.importance or 0), str(m.created_at or "")))
        self.memories = self.memories[-self.max_store_items :]

    def get_all_records(self) -> List[MemoryRecord]:
        return list(self.memories)
