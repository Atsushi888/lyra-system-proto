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

    # v1.3+（後方互換：既存JSONには無くてもOK）
    record_type: Optional[str] = None  # "world_change" / "relationship_trace" / "setting"


class MemoryAI:
    """
    長期記憶管理（v1.3 / relationship_trace 追加）

    目的：
    - world_change（importance=5）は従来通り最優先で保存
    - それ以外は「全部importance=4で保存」をやめる
    - “関係性の節目（小～中）” を relationship_trace（importance=2）として保存する
      例：外泊・初めての二人きり・強い信頼表明・告白に近い発言 etc.

    互換：
    - AnswerTalker の旧シグネチャ呼び出しにも耐える（llm_manager / model_name）
    - 既存の保存JSON（record_type無し）も読める
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
                        record_type=d.get("record_type"),  # v1.3+（無くてもOK）
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
        # v1.3+: 追加情報（呼び出し側が渡せるなら精度が上がる。渡されなくても動く）
        emotion: Optional[Dict[str, Any]] = None,
        world_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        保存ポリシー（v1.3）：
        1) world_change を検出したら importance=5 で保存（従来通り）
        2) そうでなければ relationship_trace を検出したら importance=2 で保存
        3) どちらでもなければ原則 skip（＝「設定」乱発をやめる）

        ※「設定」保存を続けたい場合は、ここに別条件で importance=4 を復活させる。
        """

        # --- 入力整形 ---
        user_text = self._extract_last_user(messages)
        assistant_text = (final_reply or "").strip()

        if not user_text and not assistant_text:
            return {"status": "skip", "added": 0}

        # --- 1) world_change（従来） ---
        wc = self._detector.detect(messages, final_reply)
        if bool(wc.get("is_world_change")):
            rec = self._build_world_change_record(
                messages=messages,
                final_reply=final_reply,
                round_id=round_id,
                user_text=user_text,
                wc=wc,
            )
            self.memories.append(rec)
            self._trim()
            self.save()
            return {"status": "ok", "added": 1, "importance": 5, "type": "world_change"}

        # --- 2) relationship_trace（新規） ---
        trace = self._detect_relationship_trace(
            messages=messages,
            final_reply=final_reply,
            emotion=emotion,
            world_state=world_state,
        )
        if trace is not None:
            rec = self._build_relationship_trace_record(
                trace=trace,
                messages=messages,
                final_reply=final_reply,
                round_id=round_id,
                user_text=user_text,
            )
            self.memories.append(rec)
            self._trim()
            self.save()
            return {"status": "ok", "added": 1, "importance": rec.importance, "type": "relationship_trace"}

        # --- 3) それ以外は保存しない（重要：ここで“記憶”がログ化するのを止める） ---
        return {"status": "skip", "added": 0}

    # ---------------- builders ----------------

    def _build_world_change_record(
        self,
        *,
        messages: List[Dict[str, Any]],
        final_reply: str,
        round_id: int,
        user_text: str,
        wc: Dict[str, Any],
    ) -> MemoryRecord:
        created_at = datetime.now(timezone.utc).isoformat()
        mem_id = f"{created_at}_{int(round_id)}"

        base_text = (final_reply or "").strip() or (user_text or "").strip()
        summary = base_text[:160] + ("…" if len(base_text) > 160 else "")

        rec = MemoryRecord(
            id=mem_id,
            round_id=int(round_id),
            importance=5,
            summary=summary,
            tags=["世界変化"],
            created_at=created_at,
            source_user=user_text,
            source_assistant=(final_reply or ""),
            record_type="world_change",
        )

        reasons = wc.get("reasons") or []
        if isinstance(reasons, list) and reasons:
            rec.world_change_reasons = [str(x) for x in reasons][:8]
        else:
            rec.reason_unavailable = self._reason_classifier.classify(
                messages=[
                    {"role": str(m.get("role")), "content": str(m.get("content", ""))}
                    for m in (messages or [])
                    if isinstance(m, dict)
                ],
                final_reply=final_reply,
            )

        return rec

    def _build_relationship_trace_record(
        self,
        *,
        trace: Dict[str, Any],
        messages: List[Dict[str, Any]],
        final_reply: str,
        round_id: int,
        user_text: str,
    ) -> MemoryRecord:
        created_at = datetime.now(timezone.utc).isoformat()
        mem_id = f"{created_at}_{int(round_id)}"

        summary = str(trace.get("summary") or "").strip()
        if not summary:
            # 念のため：空ならユーザ or 最終返答の頭を入れて落とさない
            base_text = (final_reply or "").strip() or (user_text or "").strip()
            summary = base_text[:160] + ("…" if len(base_text) > 160 else "")

        tags = trace.get("tags")
        if not isinstance(tags, list):
            tags = []
        tags = [str(x) for x in tags if str(x).strip()]

        importance = trace.get("importance", 2)
        try:
            importance_i = int(importance)
        except Exception:
            importance_i = 2
        # relationship_trace は 1〜3 に制限（ログ化させない）
        if importance_i < 1:
            importance_i = 1
        if importance_i > 3:
            importance_i = 3

        return MemoryRecord(
            id=mem_id,
            round_id=int(round_id),
            importance=importance_i,
            summary=summary[:200] + ("…" if len(summary) > 200 else ""),
            tags=tags or ["関係変化"],
            created_at=created_at,
            source_user=user_text,
            source_assistant=(final_reply or ""),
            record_type="relationship_trace",
        )

    # ---------------- relationship_trace detector ----------------

    def _detect_relationship_trace(
        self,
        *,
        messages: List[Dict[str, Any]],
        final_reply: str,
        emotion: Optional[Dict[str, Any]] = None,
        world_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        まずはルールベース（“拾う” ことを優先）。
        後でLLM要約に差し替える余地は残す。

        例：外泊・初めての二人きり・強い信頼表明など
        """

        blob = self._build_text_blob(messages=messages, final_reply=final_reply)

        # --- 外泊/泊まり：今回の件の主対象 ---
        overnight_keywords = [
            "外泊", "泊", "泊ま", "泊め", "家に", "家で", "昨夜", "朝になっ",
            "遅くな", "雨", "帰れなく", "終電", "一晩",
        ]
        hit_overnight = any(k in blob for k in overnight_keywords)

        if not hit_overnight:
            return None

        # --- emotion gate（あれば精度UP / 無ければ通す） ---
        aff = self._safe_float(emotion.get("affection") if isinstance(emotion, dict) else None, default=None)
        ten = self._safe_float(emotion.get("tension") if isinstance(emotion, dict) else None, default=None)

        # affection が明確に低いなら見送り（あれば）
        if aff is not None and aff < 0.55:
            return None

        # --- 要約（意味だけ残す） ---
        # world_state があれば「雨」「遅くなった」など補助できるが、ここでは固定でも十分
        summary = (
            "彼女は外泊という小さな一線を越え、"
            "『この人のそばなら大丈夫だ』という安心と信頼を得た。"
            "何も起きなかったこと自体が、静かな関係の前進として残った。"
        )

        tags = [
            "relationship_trace",
            "外泊",
            "信頼",
            "安心",
            "何もなかった",
        ]

        # 緊張が高いなら少し重みを上げる（任意）
        importance = 2
        if ten is not None and ten >= 0.45:
            importance = 3

        return {
            "summary": summary,
            "importance": importance,
            "tags": tags,
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
    def _build_text_blob(*, messages: List[Dict[str, Any]], final_reply: str) -> str:
        parts: List[str] = []
        for m in messages or []:
            if not isinstance(m, dict):
                continue
            r = m.get("role")
            if r not in ("user", "assistant"):
                continue
            c = str(m.get("content") or "")
            if c.strip():
                parts.append(c.strip())
        fr = (final_reply or "").strip()
        if fr:
            parts.append(fr)
        return "\n".join(parts)

    @staticmethod
    def _safe_float(v: Any, default: Optional[float] = 0.0) -> Optional[float]:
        if v is None:
            return default
        try:
            return float(v)
        except Exception:
            return default

    def _trim(self) -> None:
        if len(self.memories) <= self.max_store_items:
            return
        # 重要度が高いほど残す、同重要度なら新しいほど残す
        self.memories.sort(key=lambda m: (int(getattr(m, "importance", 0)), str(getattr(m, "created_at", ""))))
        self.memories = self.memories[-self.max_store_items :]

    def get_all_records(self) -> List[MemoryRecord]:
        return list(self.memories)
