from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from llm.llm_manager import LLMManager


@dataclass
class MemoryRecord:
    """
    長期記憶 1 件分の構造。
    """
    id: str
    round_id: int
    importance: int
    summary: str
    tags: List[str]
    created_at: str
    source_user: str
    source_assistant: str


class MemoryAI:
    """
    Lyra-System 用の記憶管理クラス（JSON 永続化版）。

    役割:
      - 1ターンの会話（ユーザー発話 + 最終返答）から、
        長期記憶にすべき内容を LLM に判定させて保存
      - 次ターンで参照するための「記憶コンテキスト」を生成
      - JSON 永続化
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        persona_id: str = "default",
        base_dir: str = "data/memory",
        model_name: str = "gpt51",
        max_store_items: int = 200,
        temperature: float = 0.2,
        max_tokens: int = 400,
    ) -> None:

        self.llm_manager = llm_manager
        self.persona_id = persona_id
        self.base_dir = base_dir
        self.model_name = model_name
        self.max_store_items = max_store_items
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        os.makedirs(self.base_dir, exist_ok=True)
        self.file_path = os.path.join(self.base_dir, f"{self.persona_id}.json")

        self.memories: List[MemoryRecord] = []
        self.load()

    # ============================
    # 永続化
    # ============================
    def load(self) -> None:
        if not os.path.exists(self.file_path):
            self.memories = []
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            self.memories = []
            return

        mems: List[MemoryRecord] = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                try:
                    mems.append(
                        MemoryRecord(
                            id=str(item.get("id", "")),
                            round_id=int(item.get("round_id", 0)),
                            importance=int(item.get("importance", 1)),
                            summary=str(item.get("summary", "")),
                            tags=list(item.get("tags", []))
                            if isinstance(item.get("tags"), list)
                            else [],
                            created_at=str(item.get("created_at", "")),
                            source_user=str(item.get("source_user", "")),
                            source_assistant=str(item.get("source_assistant", "")),
                        )
                    )
                except Exception:
                    continue

        self.memories = mems

    def save(self) -> None:
        data = [asdict(m) for m in self.memories]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ============================
    # 外部 API（ビュー用）
    # ============================
    def get_all_records(self) -> List[MemoryRecord]:
        return list(self.memories)

    # ============================
    # 1ターンの記憶更新
    # ============================
    def update_from_turn(
        self,
        messages: List[Dict[str, str]],
        final_reply: str,
        round_id: int,
    ) -> Dict[str, Any]:

        user_text = self._extract_last_user_content(messages)

        if not user_text and not final_reply:
            return {
                "status": "skip",
                "added": 0,
                "total": len(self.memories),
                "reason": "empty_turn",
                "raw_reply": "",
                "records": [],
                "error": None,
            }

        prompt = self._build_update_prompt(user_text, final_reply)
        reply_text = self._call_model_with_prompt(prompt)

        if not reply_text:
            return {
                "status": "skip",
                "added": 0,
                "total": len(self.memories),
                "reason": "no_reply",
                "raw_reply": "",
                "records": [],
                "error": None,
            }

        # JSONパース
        try:
            parsed = json.loads(reply_text)
        except Exception:
            parsed = None

        items = None
        reason = ""
        if isinstance(parsed, dict):
            reason = str(parsed.get("reason", ""))
            items = parsed.get("memories")

        added_records: List[MemoryRecord] = []

        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue

                summary = str(item.get("summary", "")).strip()
                if not summary:
                    continue

                try:
                    importance_raw = int(item.get("importance", 1))
                except Exception:
                    importance_raw = 1
                importance = max(1, min(5, importance_raw))

                tags_val = item.get("tags") or []
                if not isinstance(tags_val, list):
                    tags_val = []
                tags = [str(t) for t in tags_val]

                created_at = datetime.now(timezone.utc).isoformat()
                mem_id = f"{created_at}_{round_id}"

                rec = MemoryRecord(
                    id=mem_id,
                    round_id=round_id,
                    importance=importance,
                    summary=summary,
                    tags=tags,
                    created_at=created_at,
                    source_user=user_text,
                    source_assistant=final_reply,
                )
                self.memories.append(rec)
                added_records.append(rec)

            # 間引き → 保存
            self._trim_memories(self.max_store_items)
            self.save()

        status = "ok" if added_records else "skip"
        return {
            "status": status,
            "added": len(added_records),
            "total": len(self.memories),
            "reason": reason,
            "raw_reply": reply_text,
            "records": [asdict(r) for r in added_records],
            "error": None,
        }

    # ============================
    # コンテキスト生成（次のターン用）
    # ============================
    def build_memory_context(
        self,
        user_query: str,
        max_items: int = 5,
    ) -> str:

        if not self.memories:
            return ""

        sorted_mems = sorted(
            self.memories,
            key=lambda m: (m.importance, m.created_at),
            reverse=True,
        )

        picked = sorted_mems[:max_items]
        if not picked:
            return ""

        lines = [f"- {m.summary}" for m in picked]

        return "これまでに覚えている大切なこと:\n" + "\n".join(lines)

    # ============================
    # 内部ユーティリティ
    # ============================
    def _call_model_with_prompt(self, prompt: str) -> str:
        """
        LLMManager と model_name を使って、内部的に LLM を 1 回呼び出す。
        Memory 抽出専用の小さなユーティリティ。
        """
        messages = [{"role": "user", "content": prompt}]

        try:
            # ★ここを修正：name= ではなく model_name を第1引数に渡す
            raw = self.llm_manager.call_model(
                self.model_name,
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            # デバッグ用ログ（必要なら st.write に差し替えてもOK）
            print(f"[MemoryAI] call_model error: {e}")
            return ""

        # call_model は (reply_text, usage) or 文字列を返す設計
        if isinstance(raw, tuple) and raw:
            reply_text = raw[0]
        else:
            reply_text = raw

        if not isinstance(reply_text, str):
            return ""

        return reply_text or ""
    
    @staticmethod
    def _build_update_prompt(user_text: str, final_reply: str) -> str:
        return f"""
あなたは「長期記憶フィルタ」です。
以下の会話ログから、このキャラクターが今後も覚えておくべき
重要な出来事だけを抽出してください。

出力は必ず JSON のみ。

user: {user_text}
assistant: {final_reply}

JSON形式:
{{
  "reason": "理由",
  "memories": [
    {{
      "summary": "短い要約",
      "importance": 1,
      "tags": ["関係性", "設定"]
    }}
  ]
}}
""".strip()

    @staticmethod
    def _extract_last_user_content(messages: List[Dict[str, Any]]) -> str:
        if not isinstance(messages, list):
            return ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return str(msg.get("content", ""))
        return ""

    def _trim_memories(self, max_items: int = 200) -> None:
        if len(self.memories) <= max_items:
            return

        sorted_mems = sorted(
            self.memories,
            key=lambda m: (m.importance, m.created_at),
            reverse=False,
        )

        self.memories = sorted_mems[-max_items:]
