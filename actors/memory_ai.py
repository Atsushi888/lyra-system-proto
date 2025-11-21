# actors/memory_ai.py

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

    - id:        一意なID（作成時刻 + ラウンド番号など）
    - round_id:  会話ログ上のラウンド番号
    - importance:重要度 1〜5（5が最重要）
    - summary:   記憶の要約（日本語）
    - tags:      "関係性", "感情", "設定" などのタグ
    - created_at:ISO8601 形式の作成日時（UTC）
    - source_user:      ユーザー発話（元テキスト）
    - source_assistant: アシスタントの最終返答（元テキスト）
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
    Lyra-System 用の記憶管理クラス（v0.1 / JSON 永続化版）。

    役割:
      - 1ターンの会話（ユーザー発話 + 最終返答）から、
        「長期記憶に残すべき内容」があるかどうかを LLM に判定させる
      - 必要な場合、その要約を MemoryRecord として保存する
      - 次のターン用に「関連しそうな記憶のまとめテキスト」を返す
      - 記憶は JSON ファイルとして永続化する

    特徴:
      - importance (1〜5) × 新しさ でソートして利用
      - v0.1 ではまだベクトル検索は行わない（シンプル優先）
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
        """
        Parameters
        ----------
        llm_manager:
            LLMManager インスタンス。
            内部で LLMRouter を保持している前提。

        persona_id:
            記憶ファイルを分けるための ID。
            例: Persona.char_id ("floria_ja" など)

        base_dir:
            記憶ファイルを保存するディレクトリ。

        model_name:
            記憶抽出に使うモデル名。 "gpt51" / "gpt4o" / "hermes" など。

        max_store_items:
            記憶の最大保持件数。超えた分は低重要度・古いものから削除する。
        """
        self.llm_manager = llm_manager
        self.persona_id = persona_id
        self.base_dir = base_dir
        self.model_name = model_name
        self.max_store_items = max_store_items
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        os.makedirs(self.base_dir, exist_ok=True)
        self.file_path = os.path.join(self.base_dir, f"{self.persona_id}.json")

        # メモリ本体
        self.memories: List[MemoryRecord] = []
        self.load()

    # ============================
    # 永続化
    # ============================
    def load(self) -> None:
        """
        JSON ファイルから記憶を読み込む。
        無ければ空リストのまま。
        """
        if not os.path.exists(self.file_path):
            self.memories = []
            return

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            # 壊れていたら空からやり直す
            self.memories = []
            return

        mems: List[MemoryRecord] = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                try:
                    mem = MemoryRecord(
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
                    mems.append(mem)
                except Exception:
                    # 1件壊れていても他は読み込む
                    continue

        self.memories = mems

    def save(self) -> None:
        """
        現在の記憶を JSON として保存する。
        """
        data = [asdict(m) for m in self.memories]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


    def save(self) -> None:
        """
        現在の記憶を JSON として保存する。
        """
        data = [asdict(m) for m in self.memories]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
        # デバッグ用ログ
        print(
            f"[MemoryAI.save] persona_id={self.persona_id}, "
            f"records={len(self.memories)}, path={self.file_path}"
        )
    

    # ============================
    # 公開: 記憶一覧取得（ビュー用）
    # ============================
    def get_all_records(self) -> List[MemoryRecord]:
        """
        デバッグビュー用：現在保持している MemoryRecord をそのまま返す。
        """
        return list(self.memories)

    # ============================
    # 公開: 記憶更新
    # ============================
    def update_from_turn(
        self,
        messages: List[Dict[str, str]],
        final_reply: str,
        round_id: int,
    ) -> Dict[str, Any]:
        """
        1ターン分の会話から、長期記憶に残すべき内容があれば追加する。

        Parameters
        ----------
        messages:
            Persona.build_messages(user_text) で組んだ messages。
            少なくとも最後に user ロールが含まれていることを想定。

        final_reply:
            このターンの最終返答（ComposerAI の結果など）。

        round_id:
            このターンのラウンド番号（ログ上の通し番号）。

        Returns
        -------
        Dict[str, Any]:
            {
              "status": "ok" | "skip" | "error",
              "added": int,
              "total": int,
              "reason": str,
              "raw_reply": str,
              "records": [MemoryRecord ... を dict 化したもの],
              "error": Optional[str],
            }
        """
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

        # 記憶抽出用 LLM 呼び出し
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

        added_records: List[MemoryRecord] = []
        reason = ""
        error: Optional[str] = None

        # JSON パース（壊れていても落ちないように）
        try:
            parsed = json.loads(reply_text)
        except Exception:
            parsed = None

        items = None
        if isinstance(parsed, dict):
            reason = str(parsed.get("reason", ""))
            items = parsed.get("memories")

        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue

                summary = str(item.get("summary", "")).strip()
                if not summary:
                    continue

                # importance は 1〜5 にクリップ
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

            # importance が低いものから順に間引く
            self._trim_memories(max_items=self.max_store_items)
            # 保存
            self.save()

        status = "ok" if added_records else "skip"

        return {
            "status": status,
            "added": len(added_records),
            "total": len(self.memories),
            "reason": reason,
            "raw_reply": reply_text,
            "records": [asdict(r) for r in added_records],
            "error": error,
        }

    # ============================
    # 公開: コンテキスト構築
    # ============================
    def build_memory_context(
        self,
        user_query: str,
        max_items: int = 5,
    ) -> str:
        """
        次のターンの system に差し込む用の「記憶コンテキスト」を組み立てる。

        v0.1 では:
          - importance の高い順
          - 同じ importance 内では新しい順
        で最大 max_items 件を取り出し、箇条書きテキストにまとめる。
        """
        if not self.memories:
            return ""

        # importance / created_at でソート
        sorted_mems = sorted(
            self.memories,
            key=lambda m: (m.importance, m.created_at),
            reverse=True,
        )

        picked = sorted_mems[:max_items]
        if not picked:
            return ""

        lines: List[str] = []
        for m in picked:
            lines.append(f"- {m.summary}")

        # ここはペルソナ依存の文言でもよいが、汎用的に記述
        context = "これまでに覚えている大切なこと:\n" + "\n".join(lines)
        return context

    # ============================
    # 内部: LLM 呼び出し
    # ============================
    def _call_model_with_prompt(self, prompt: str) -> str:
        """
        LLMManager と model_name を使って、内部的に LLM を 1 回呼び出す。
        Memory 抽出専用の小さなユーティリティ。
        """
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = self.llm_manager.call_model(
                name=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception:
            return ""

        # call_model は (reply_text, usage) を返す設計
        if isinstance(raw, tuple) and raw:
            reply_text = raw[0]
        else:
            reply_text = raw

        if not isinstance(reply_text, str):
            return ""

        return reply_text or ""

    # ============================
    # 内部: プロンプト構築
    # ============================
    @staticmethod
    def _build_update_prompt(user_text: str, final_reply: str) -> str:
        """
        このターンからどのような記憶を残すべきかを判定させるためのプロンプト。
        LLM から JSON を返させる。
        """
        return f"""
あなたは「長期記憶フィルタ」です。
以下の会話ログから、このキャラクターが今後も覚えておくべき
「重要な出来事・設定・感情の変化」だけを抽出してください。

出力は必ず JSON 形式で、日本語のままにしてください。

[入力フォーマット]
- user: プレイヤーの発話
- assistant: キャラクターの返答

[会話]
user: {user_text}
assistant: {final_reply}

[出力フォーマット（JSON のみ）]
{{
  "reason": "どのような観点で記憶を選んだかの簡単な説明（日本語）",
  "memories": [
    {{
      "summary": "保存すべき内容の短い要約（日本語）",
      "importance": 1〜5 の整数（5 が最重要）,
      "tags": ["関係性", "感情", "設定"] のような日本語タグの配列
    }},
    ...
  ]
}}

条件:
- 本当に何も記憶する必要がない場合、"memories" は空配列にしてください。
- 冗長な日常会話や一時的な話題（あいさつ、ちょっとした感想など）は保存しないでください。
- プレイヤーとの関係性の変化、今回初めて出てきた重要な情報、
  今後の物語に関わりそうな約束などは優先的に保存してください。
""".strip()

    # ============================
    # 内部: 補助
    # ============================
    @staticmethod
    def _extract_last_user_content(messages: List[Dict[str, Any]]) -> str:
        """
        messages から最後の user メッセージの content を抽出。
        見つからなければ空文字。
        """
        if not isinstance(messages, list):
            return ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
        return ""

    def _trim_memories(self, max_items: int = 200) -> None:
        """
        記憶数が max_items を超えた場合、
        importance が低く古いものから削除する。
        """
        if len(self.memories) <= max_items:
            return

        # importance 昇順 / created_at 昇順 = 「低重要度かつ古いもの」が前
        sorted_mems = sorted(
            self.memories,
            key=lambda m: (m.importance, m.created_at),
            reverse=False,
        )

        # 後ろ max_items 件だけ残す
        keep = sorted_mems[-max_items:]
        self.memories = keep
