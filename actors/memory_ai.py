# actors/memory_ai.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# World change detector / reason classifier
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
    - AnswerTalker 側の古い呼び出し（llm_manager/model_name/temperature/max_tokens 等）でも落ちない
    - update_from_turn は positional / keyword の双方に対応
    """

    def __init__(
        self,
        llm_manager: Any = None,  # 後方互換（v1.1では未使用）
        *,
        persona_id: str = "default",
        persona_raw: Optional[Dict[str, Any]] = None,
        base_dir: str = "data/memory",
        max_store_items: int = 200,
        # 後方互換（未使用だが受け取る）
        model_name: str = "gpt4o",
        temperature: float = 0.2,
        max_tokens: int = 400,
        **_ignored: Any,  # 将来拡張で渡される未知引数は握り潰す
    ) -> None:
        self.llm_manager = llm_manager  # 互換のため保持（未使用）
        self.persona_id = str(persona_id or "default")
        self.persona_raw = persona_raw or {}
        self.max_store_items = int(max_store_items)

        # 互換パラメータ（現状未使用。デバッグ表示や将来復活に備えて保持）
        self.model_name = str(model_name)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        os.makedirs(base_dir, exist_ok=True)
        self.file_path = os.path.join(base_dir, f"{self.persona_id}.json")

        self.memories: List[MemoryRecord] = []

        self._detector = WorldChangeDetector(self.persona_raw)
        self._reason_classifier = WorldChangeReasonClassifier(persona_id=self.persona_id)

        self.load()

    # ---------------- persistence ----------------

    def load(self) -> None:
        if not os.path.exists(self.file_path):
            self.memories = []
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # 壊れていたら空から
            self.memories = []
            return

        if not isinstance(data, list):
            self.memories = []
            return

        mems: List[MemoryRecord] = []
        for d in data:
            if not isinstance(d, dict):
                continue
            try:
                mems.append(
                    MemoryRecord(
                        id=str(d.get("id", "")),
                        round_id=int(d.get("round_id", 0) or 0),
                        importance=int(d.get("importance", 1) or 1),
                        summary=str(d.get("summary", "") or ""),
                        tags=list(d.get("tags", []) or []),
                        created_at=str(d.get("created_at", "") or ""),
                        source_user=str(d.get("source_user", "") or ""),
                        source_assistant=str(d.get("source_assistant", "") or ""),
                        world_change_reasons=d.get("world_change_reasons"),
                        reason_unavailable=d.get("reason_unavailable"),
                    )
                )
            except Exception:
                continue

        self.memories = mems

    def save(self) -> None:
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(
                    [asdict(m) for m in self.memories],
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            # 記憶保存失敗で会話全体が落ちるのは避ける
            return

    # ---------------- main ----------------

    def update_from_turn(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        final_reply: str = "",
        round_id: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        1ターン分から MemoryRecord を生成して保存。

        後方互換：
        - update_from_turn(messages, final_reply, round_id) の positional 呼び出し
        - update_from_turn(messages=..., final_reply=..., round_id=...) の keyword 呼び出し
        - 余計な kwargs（例: persona_id 等）が来ても落とさない
        """
        # keyword で渡されていたらそちらを優先（混在事故の防止）
        if messages is None:
            messages = kwargs.get("messages") or []
        if not final_reply:
            final_reply = str(kwargs.get("final_reply") or "")
        if not round_id:
            try:
                round_id = int(kwargs.get("round_id") or 0)
            except Exception:
                round_id = 0

        messages = messages or []
        final_reply = str(final_reply or "")

        # world change 判定（先に）
        wc = self._detector.detect(messages, final_reply)

        user_text = self._extract_last_user(messages)
        base_text = (final_reply or "").strip() or (user_text or "").strip()

        if not base_text:
            return {"status": "skip", "added": 0, "reason": "empty_turn"}

        created_at = datetime.now(timezone.utc).isoformat()
        mem_id = f"{created_at}_{int(round_id)}"

        importance = 5 if bool(wc.get("is_world_change")) else 4

        # summary: 160文字でトリム
        max_summary_len = 160
        summary = base_text
        if len(summary) > max_summary_len:
            summary = summary[: max_summary_len - 1] + "…"

        rec = MemoryRecord(
            id=mem_id,
            round_id=int(round_id),
            importance=int(importance),
            summary=summary,
            tags=["世界変化"] if importance >= 5 else ["設定"],
            created_at=created_at,
            source_user=user_text,
            source_assistant=final_reply,
        )

        # importance=5 の場合は reasons を埋める（無ければ分類器）
        if importance >= 5:
            reasons = wc.get("reasons")
            if isinstance(reasons, list) and reasons:
                rec.world_change_reasons = [str(x) for x in reasons if str(x).strip()]
                if not rec.world_change_reasons:
                    rec.world_change_reasons = None

            if rec.world_change_reasons is None:
                try:
                    rec.reason_unavailable = self._reason_classifier.classify(
                        messages, final_reply
                    )
                except Exception:
                    rec.reason_unavailable = "unknown"

        self.memories.append(rec)
        self._trim()
        self.save()

        return {
            "status": "ok",
            "added": 1,
            "importance": importance,
            "id": rec.id,
            "tags": rec.tags,
            "world_change": bool(importance >= 5),
        }

    # ---------------- helpers ----------------

    @staticmethod
    def _extract_last_user(messages: List[Dict[str, Any]]) -> str:
        for m in reversed(messages or []):
            try:
                if isinstance(m, dict) and m.get("role") == "user":
                    return str(m.get("content") or "")
            except Exception:
                continue
        return ""

    def _trim(self) -> None:
        if len(self.memories) <= self.max_store_items:
            return
        # importance 昇順 / created_at 昇順 => 「低重要度かつ古い」が前
        self.memories.sort(key=lambda m: (m.importance, m.created_at))
        self.memories = self.memories[-self.max_store_items :]

    def get_all_records(self) -> List[MemoryRecord]:
        return list(self.memories)

    def build_memory_context(self, max_items: int = 5) -> str:
        """
        次ターンに差し込むための簡易コンテキスト。
        importance 高い順＋新しい順で max_items 件。
        """
        if not self.memories:
            return ""

        sorted_mems = sorted(
            self.memories,
            key=lambda m: (m.importance, m.created_at),
            reverse=True,
        )
        picked = sorted_mems[: max(0, int(max_items))]

        if not picked:
            return ""

        lines: List[str] = []
        for m in picked:
            lines.append(f"- {m.summary}")
        return "これまでに覚えている大切なこと:\n" + "\n".join(lines)
