# actors/memory_ai.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from actors.persona.world_change_detector import WorldChangeDetector
from actors.memory.world_change_reason_classifier import WorldChangeReasonClassifier
from actors.memory.memory_importance_classifier import MemoryImportanceClassifier

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

    # v1.3+（任意：重要度AIのログを残したい場合）
    importance_model: Optional[str] = None
    importance_debug: Optional[Dict[str, Any]] = None


class MemoryAI:
    """
    長期記憶管理（v1.3 / 重要度判定=他AI単発）

    - importance=5（世界変化）を最優先で扱う（従来通り）
    - 世界変化でない場合でも、「イベント系キーワード」にヒットしたら
      他AI（単発）で importance(1..4) / summary / tags を判定する
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
        preferred_importance_model: Optional[str] = None,
    ) -> None:
        self.persona_id = str(persona_id or "default")
        self.persona_raw = persona_raw or {}
        self.max_store_items = int(max_store_items)

        os.makedirs(base_dir, exist_ok=True)
        self.file_path = os.path.join(base_dir, f"{self.persona_id}.json")

        self.memories: List[MemoryRecord] = []

        # 世界変化検出（importance=5）
        self._detector = WorldChangeDetector(self.persona_raw)

        # 世界変化理由の2択分類（reasonsが取れない時だけ）
        self._reason_classifier = WorldChangeReasonClassifier(
            persona_id=self.persona_id,
            preferred_model=(preferred_reason_model or model_name or "gpt52"),
            llm_manager=llm_manager,
        )

        # ★追加：重要度(1..4)判定（他AI単発）
        self._importance_classifier = MemoryImportanceClassifier(
            persona_id=self.persona_id,
            preferred_model=(preferred_importance_model or model_name or "gpt52"),
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
                        importance_model=d.get("importance_model"),
                        importance_debug=d.get("importance_debug"),
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

        # ---------------- importance決定 ----------------
        if bool(wc.get("is_world_change")):
            importance = 5
            summary = base_text[:160] + ("…" if len(base_text) > 160 else "")
            tags = ["世界変化"]
            importance_model = None
            importance_debug = None
        else:
            # ★世界変化ではない場合：「イベント系キーワード」ヒットがあれば他AIで判定
            hit = self._detect_memory_event_keywords(user_text=user_text, final_reply=final_reply)

            if hit:
                # 他AI単発で importance(1..4), summary, tags を決める
                cls = self._importance_classifier.classify(
                    messages=self._normalize_messages_for_classifier(messages),
                    final_reply=final_reply,
                    event_keywords_hit=hit,
                )
                importance = int(cls.get("importance", 3))
                summary = str(cls.get("summary") or "").strip() or (base_text[:160] + ("…" if len(base_text) > 160 else ""))
                tags_raw = cls.get("tags")
                tags = [str(x) for x in tags_raw] if isinstance(tags_raw, list) else ["イベント"]
                importance_model = str(cls.get("model") or "")
                # デバッグは重いので必要最小限（raw_textは残してOK）
                importance_debug = {
                    "status": cls.get("status"),
                    "model": cls.get("model"),
                    "raw_text": cls.get("raw_text"),
                    "error": cls.get("error"),
                    "keywords_hit": hit,
                }
            else:
                # 従来互換：世界変化ではないが、保存するなら設定扱い（importance=4）
                # （ここを「skip」にしたい場合は、この else を return skip に変える）
                importance = 4
                summary = base_text[:160] + ("…" if len(base_text) > 160 else "")
                tags = ["設定"]
                importance_model = None
                importance_debug = None

        rec = MemoryRecord(
            id=mem_id,
            round_id=int(round_id),
            importance=int(importance),
            summary=summary,
            tags=tags[:8],
            created_at=created_at,
            source_user=user_text,
            source_assistant=(final_reply or ""),
            importance_model=importance_model,
            importance_debug=importance_debug,
        )

        # ---------------- 世界変化理由の付与（importance=5のみ） ----------------
        if int(importance) == 5:
            reasons = wc.get("reasons") or []
            if isinstance(reasons, list) and reasons:
                rec.world_change_reasons = [str(x) for x in reasons][:8]
            else:
                rec.reason_unavailable = self._reason_classifier.classify(
                    messages=self._normalize_messages_for_classifier(messages),
                    final_reply=final_reply,
                )

        self.memories.append(rec)
        self._trim()
        self.save()

        return {
            "status": "ok",
            "added": 1,
            "importance": int(importance),
            "tags": rec.tags,
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

    @staticmethod
    def _normalize_messages_for_classifier(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for m in messages or []:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or "")
            content = str(m.get("content") or "")
            if role and content:
                out.append({"role": role, "content": content})
        return out

    def _trim(self) -> None:
        if len(self.memories) <= self.max_store_items:
            return
        # 重要度が高いほど残す、同重要度なら新しいほど残す
        self.memories.sort(
            key=lambda m: (int(getattr(m, "importance", 0)), str(getattr(m, "created_at", "")))
        )
        self.memories = self.memories[-self.max_store_items :]

    def get_all_records(self) -> List[MemoryRecord]:
        return list(self.memories)

    # ---------------- memory-event keyword detection ----------------

    def _detect_memory_event_keywords(self, *, user_text: str, final_reply: str) -> List[str]:
        """
        「世界変化ではないが記憶に残すべきイベント」を拾うための軽量キーワード検出。

        - persona_raw["memory_event_keywords"] があれば追加
        - デフォルトは「外泊」を確実に拾える語を入れておく
        """
        defaults = [
            "外泊",
            "泊ま",
            "お泊まり",
            "宿泊",
            "帰れない",
            "終電",
            "夜を明か",
            "泊まり",
        ]

        extra = self.persona_raw.get("memory_event_keywords", [])
        extra_list: List[str] = [str(x) for x in extra] if isinstance(extra, list) else []

        keywords = list(dict.fromkeys(defaults + extra_list))  # preserve order + unique

        hay = (user_text or "").strip()
        if not hay:
            hay = ""
        fr = (final_reply or "").strip()

        hit: List[str] = []
        for k in keywords:
            if not k:
                continue
            if (k in hay) or (k in fr):
                hit.append(k)

        return hit[:8]
