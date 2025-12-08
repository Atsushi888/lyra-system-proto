# actors/memory_ai.py

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from llm.llm_manager import LLMManager  # インターフェースは残すが、内部では使わない


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
    - source_assistant: アシスタントの最終返答（元テキスト＝ComposerAI出力）
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
    Lyra-System 用の記憶管理クラス（v0.2 / ComposerAI 直結版）。

    役割:
      - 1ターンの会話（ユーザー発話 + ComposerAI の最終返答）から、
        「長期記憶として残しておくべきスナップショット」を生成する
      - 生成した MemoryRecord を JSON ファイルとして永続化する
      - 次のターン用に「重要そうな記憶のまとめテキスト」を返す

    特徴:
      - もはや内部で LLM を呼び出さない（安定性と速度を優先）
      - 「どのテキストを記憶に使うか」は ComposerAI の最終出力に完全依存
      - importance / tags は、現段階ではシンプルな固定値ロジック
        （あとからヒューリスティックや別AIによる分類に差し替えやすい設計）
    """

    def __init__(
        self,
        llm_manager: Optional[LLMManager],  # インターフェース互換のため引数は残すが未使用
        persona_id: str = "default",
        base_dir: str = "data/memory",
        model_name: str = "gpt4o",         # 互換性のため残しているが利用しない
        max_store_items: int = 200,
        temperature: float = 0.2,          # 互換性のため残しているが利用しない
        max_tokens: int = 400,             # 同上
    ) -> None:
        """
        Parameters
        ----------
        llm_manager:
            互換性のため残しているが、v0.2 では内部で LLM 呼び出しを行わない。

        persona_id:
            記憶ファイルを分けるための ID。
            例: Persona.char_id ("floria_ja" など)

        base_dir:
            記憶ファイルを保存するディレクトリ。

        model_name / temperature / max_tokens:
            v0.1 との互換性維持のため残しているが、v0.2 では使用しない。

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

    # ============================
    # 公開: 記憶一覧取得（ビュー用）
    # ============================
    def get_all_records(self) -> List[MemoryRecord]:
        """
        デバッグビュー用：現在保持している MemoryRecord をそのまま返す。
        """
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
        """
        1ターン分の会話から、長期記憶に残す内容を生成する。

        v0.2 では:
          - 追加の LLM 呼び出しは行わない
          - ComposerAI の最終返答（final_reply）をベースに 1 件の MemoryRecord を生成
          - summary は final_reply からのシンプルなトリミング
          - importance や tags は、今は固定ロジック（将来拡張を見越して関数化）

        Parameters
        ----------
        messages:
            Persona.build_messages(user_text) で組んだ messages。
            少なくとも最後に user ロールが含まれていることを想定。

        final_reply:
            ComposerAI による最終返答テキスト。

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
              "raw_reply": str,  # v0.1 互換のため final_reply をそのまま入れておく
              "records": [MemoryRecord ... を dict 化したもの],
              "error": Optional[str],
            }
        """
        # ユーザー側の最後の発話も保存しておく
        user_text = self._extract_last_user_content(messages)

        # 何も材料がなければスキップ
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

        # summary の素材：優先順位は「final_reply > user_text」
        base_text = (final_reply or "").strip() or (user_text or "").strip()

        if not base_text:
            # 念のための二重防御
            return {
                "status": "skip",
                "added": 0,
                "total": len(self.memories),
                "reason": "no_base_text",
                "raw_reply": "",
                "records": [],
                "error": None,
            }

        # ---- ここが v0.2 の肝：summary / importance / tags をローカルで決める ----

        # summary は、長すぎると扱いづらいので先頭 160 文字程度にトリミング
        max_summary_len = 160
        if len(base_text) > max_summary_len:
            summary = base_text[: max_summary_len - 1] + "…"
        else:
            summary = base_text

        # importance は現段階では固定値（将来ここにヒューリスティックを入れる）
        importance = self._estimate_importance(user_text=user_text, final_reply=final_reply)

        # tags も現段階ではシンプルに ["設定"] など。将来拡張用のフック。
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

        # importance が低いものから順に間引く
        self._trim_memories(max_items=self.max_store_items)
        # 保存
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
    def build_memory_context(
        self,
        user_query: str,
        max_items: int = 5,
    ) -> str:
        """
        次のターンの system に差し込む用の「記憶コンテキスト」を組み立てる。

        v0.2 でもロジックは v0.1 と同じ：
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
    # 内部: 重要度＆タグの簡易推定
    # ============================
    @staticmethod
    def _estimate_importance(user_text: str, final_reply: str) -> int:
        """
        v0.2 ではシンプルなヒューリスティック。
        将来的に、別の軽量モデルやルールベースに差し替え可能。
        """
        text = (user_text or "") + " " + (final_reply or "")

        # ざっくりしたキーワードベースのブースト（必要なら調整）
        high_keywords = ["約束", "好き", "愛してる", "結婚", "子ども", "永遠", "大事", "重要"]
        low_keywords = ["おはよう", "こんにちは", "こんばんは", "おやすみ", "テスト", "試験"]

        imp = 3  # デフォルト

        if any(k in text for k in high_keywords):
            imp = 5
        elif any(k in text for k in low_keywords):
            imp = 2

        return imp

    @staticmethod
    def _estimate_tags(user_text: str, final_reply: str) -> List[str]:
        """
        v0.2 ではごく簡単なタグ付け。
        将来、タグ分類用の小さなモデルなどに差し替えやすい構造にしておく。
        """
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
