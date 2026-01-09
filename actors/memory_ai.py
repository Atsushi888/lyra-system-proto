# actors/memory_ai.py
from __future__ import annotations

import json
import os
import logging
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm.llm_manager import LLMManager  # インターフェースは残します（v0.2では未使用）

logger = logging.getLogger(__name__)


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


class MemoryAI:
    """
    Lyra-System 用の記憶管理クラス（v0.2 / ComposerAI 直結版）。

    役割:
      - 1ターン分（ユーザー発話 + 最終返答）から MemoryRecord を生成します
      - JSON ファイルへ永続化します
      - 次ターン用の memory context を組み立てます
    """

    def __init__(
        self,
        llm_manager: Optional[LLMManager],
        persona_id: str = "default",
        base_dir: str = "data/memory",
        model_name: str = "gpt4o",         # 互換性のため残します（v0.2では未使用）
        max_store_items: int = 200,
        temperature: float = 0.2,          # 同上
        max_tokens: int = 400,             # 同上
    ) -> None:
        self.llm_manager = llm_manager
        self.persona_id = persona_id
        self.base_dir = base_dir
        self.model_name = model_name
        self.max_store_items = max_store_items
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        # --- 保存先を絶対パスに固定します（相対パス罠を避けます） ---
        module_dir = Path(__file__).resolve().parent
        base_path = Path(base_dir)
        if not base_path.is_absolute():
            base_path = (module_dir / base_path).resolve()

        self.base_path: Path = base_path

        # まずは指定先を作成します
        try:
            self.base_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            # 万一作れない場合は /tmp に逃がします（Streamlit Cloud 対策）
            logger.exception("MemoryAI: base_dir mkdir failed, fallback to /tmp: %s", e)
            self.base_path = Path("/tmp/lyra_memory").resolve()
            self.base_path.mkdir(parents=True, exist_ok=True)

        self.file_path: Path = self.base_path / f"{self.persona_id}.json"

        # メモリ本体
        self.memories: List[MemoryRecord] = []
        self.load()

        logger.warning("MemoryAI initialized: persona_id=%s file_path=%s", self.persona_id, str(self.file_path))

    # ============================
    # 永続化
    # ============================
    def load(self) -> None:
        """
        JSON ファイルから記憶を読み込みます。存在しなければ空です。
        """
        if not self.file_path.exists():
            self.memories = []
            return

        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.exception("MemoryAI.load failed (reset to empty): %s", e)
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
                            tags=list(item.get("tags", [])) if isinstance(item.get("tags"), list) else [],
                            created_at=str(item.get("created_at", "")),
                            source_user=str(item.get("source_user", "")),
                            source_assistant=str(item.get("source_assistant", "")),
                        )
                    )
                except Exception:
                    continue

        self.memories = mems
        logger.warning("MemoryAI.load ok: total=%s file=%s", len(self.memories), str(self.file_path))

    def save(self) -> None:
        """
        現在の記憶を JSON として保存します（原子的に書き換えます）。
        """
        data = [asdict(m) for m in self.memories]

        try:
            # テンポラリに書いてから置換します
            self.base_path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", delete=False, dir=str(self.base_path), encoding="utf-8") as tf:
                json.dump(data, tf, ensure_ascii=False, indent=2)
                tmp_name = tf.name

            os.replace(tmp_name, str(self.file_path))
            logger.warning("MemoryAI.save ok: total=%s file=%s", len(self.memories), str(self.file_path))

        except Exception as e:
            logger.exception("MemoryAI.save failed: %s", e)
            raise

    # ============================
    # 公開: 記憶一覧取得（ビュー用）
    # ============================
    def get_all_records(self) -> List[MemoryRecord]:
        return list(self.memories)

    # ============================
    # 公開: 記憶更新（ComposerAI 出力直結）
    # ============================
    def update_from_turn(
        self,
        messages: List[Dict[str, str]],
        final_reply: str,
        round_id: int,
    ) -> Dict[str, Any]:
        # 呼ばれているかの確認ログ（重要）
        logger.warning("MemoryAI.update_from_turn called: round_id=%s user_len=%s final_len=%s",
                       round_id, len(self._extract_last_user_content(messages)), len(final_reply or ""))

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

        base_text = (final_reply or "").strip() or (user_text or "").strip()
        if not base_text:
            return {
                "status": "skip",
                "added": 0,
                "total": len(self.memories),
                "reason": "no_base_text",
                "raw_reply": "",
                "records": [],
                "error": None,
            }

        max_summary_len = 160
        summary = base_text[: max_summary_len - 1] + "…" if len(base_text) > max_summary_len else base_text

        importance = self._estimate_importance(user_text=user_text, final_reply=final_reply)
        tags = self._estimate_tags(user_text=user_text, final_reply=final_reply)

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
        self._trim_memories(max_items=self.max_store_items)

        # 保存（失敗時は例外ログが出ます）
        self.save()

        return {
            "status": "ok",
            "added": 1,
            "total": len(self.memories),
            "reason": "local_summary_v0.2",
            "raw_reply": final_reply,
            "records": [asdict(rec)],
            "error": None,
        }

    # ============================
    # 公開: コンテキスト構築
    # ============================
    def build_memory_context(self, user_query: str, max_items: int = 5) -> str:
        if not self.memories:
            return ""

        sorted_mems = sorted(self.memories, key=lambda m: (m.importance, m.created_at), reverse=True)
        picked = sorted_mems[:max_items]
        if not picked:
            return ""

        lines = [f"- {m.summary}" for m in picked]
        return "これまでに覚えている大切なこと:\n" + "\n".join(lines)

    # ============================
    # 内部: 重要度＆タグ推定
    # ============================
    @staticmethod
    def _estimate_importance(user_text: str, final_reply: str) -> int:
        text = (user_text or "") + " " + (final_reply or "")
        high_keywords = ["約束", "好き", "愛してる", "結婚", "子ども", "永遠", "大事", "重要"]
        low_keywords = ["おはよう", "こんにちは", "こんばんは", "おやすみ", "テスト", "試験"]

        imp = 3
        if any(k in text for k in high_keywords):
            imp = 5
        elif any(k in text for k in low_keywords):
            imp = 2
        return imp

    @staticmethod
    def _estimate_tags(user_text: str, final_reply: str) -> List[str]:
        text = (user_text or "") + " " + (final_reply or "")

        tags: List[str] = []
        if any(k in text for k in ["好き", "愛してる", "キス", "抱きしめ", "関係", "恋人"]):
            tags.append("関係性")
        if any(k in text for k in ["悲しい", "嬉しい", "楽しい", "寂しい", "怖い", "不安"]):
            tags.append("感情")
        if not tags:
            tags.append("設定")
        return tags

    # ============================
    # 内部: 補助
    # ============================
    @staticmethod
    def _extract_last_user_content(messages: List[Dict[str, Any]]) -> str:
        if not isinstance(messages, list):
            return ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
        return ""

    def _trim_memories(self, max_items: int = 200) -> None:
        if len(self.memories) <= max_items:
            return
        sorted_mems = sorted(self.memories, key=lambda m: (m.importance, m.created_at), reverse=False)
        self.memories = sorted_mems[-max_items:]
