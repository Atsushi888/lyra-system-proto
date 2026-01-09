# actors/memory_ai.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from actors.persona.world_change_detector import WorldChangeDetector
from actors.memory.world_change_reason_classifier import WorldChangeReasonClassifier

try:
    from llm.llm_manager import LLMManager
except Exception:  # pragma: no cover
    LLMManager = Any  # type: ignore


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

    # v1.1+
    world_change_reasons: Optional[List[str]] = None
    reason_unavailable: Optional[str] = None


class MemoryAI:
    """
    長期記憶管理（v1.2 / 互換強化）

    - importance=5（世界変化）を最優先で扱う
    - reasons が取れない場合は外部分類器で reason_unavailable を付与
    - AnswerTalker の旧シグネチャ呼び出しにも耐える（llm_manager / model_name）
    """

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,   # ★旧呼び出し互換（positional もOK）
        *,
        persona_id: str,
        persona_raw: Optional[Dict[str, Any]] = None,
        base_dir: str = "data/memory",
        max_store_items: int = 200,
        # ★旧呼び出し互換：AnswerTalker が model_name=memory_model を渡してくる
        model_name: Optional[str] = None,
        preferred_reason_model: Optional[str] = None,
    ) -> None:
        self.persona_id = str(persona_id or "default")
        self.persona_raw = persona_raw or {}
        self.max_store_items = int(max_store_items)

        os.makedirs(base_dir, exist_ok=True)
        self.file_path = os.path.join(base_dir, f"{self.persona_id}.json")

        self.memories: List[MemoryRecord] = []

        self._detector = WorldChangeDetector(self.persona_raw)

        # ★分類器：可能なら同じ llm_manager を共有（persona_id ズレや二重生成を避ける）
        self._reason_classifier = WorldChangeReasonClassifier(
            persona_id=self.persona_id,
            preferred_model=(preferred_reason_model or model_name or "gpt52"),
            llm_manager=llm_manager,
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
                        tags=list(d.get("tags", [])) if isinstance(d.get("tags", []), list) else [],
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
        mem_id = f"{created_at}_{int(round_id)}"

        importance = 5 if bool(wc.get("is_world_change")) else 4

        rec = MemoryRecord(
            id=mem_id,
            round_id=int(round_id),
            importance=int(importance),
            summary=base_text[:160] + ("…" if len(base_text) > 160 else ""),
            tags=["世界変化"] if importance == 5 else ["設定"],
            created_at=created_at,
            source_user=user_text,
            source_assistant=(final_reply or ""),
        )

        if importance == 5:
            reasons = wc.get("reasons") or []
            if isinstance(reasons, list) and reasons:
                rec.world_change_reasons = [str(x) for x in reasons][:8]
            else:
                rec.reason_unavailable = self._reason_classifier.classify(
                    messages=[{"role": str(m.get("role")), "content": str(m.get("content", ""))} for m in (messages or []) if isinstance(m, dict)],
                    final_reply=final_reply,
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
    def _extract_last_user(messages: Sequence[Dict[str, Any]]) -> str:
        for m in reversed(list(messages or [])):
            try:
                if m.get("role") == "user":
                    return str(m.get("content") or "")
            except Exception:
                continue
        return ""

    def _trim(self) -> None:
        if len(self.memories) <= self.max_store_items:
            return
        # 重要度が高いほど残す、同重要度なら新しいほど残す
        self.memories.sort(key=lambda m: (int(getattr(m, "importance", 0)), str(getattr(m, "created_at", ""))))
        self.memories = self.memories[-self.max_store_items :]

    def get_all_records(self) -> List[MemoryRecord]:
        return list(self.memories)
