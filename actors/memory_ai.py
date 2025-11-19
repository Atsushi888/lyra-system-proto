# actors/memory_ai.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import streamlit as st

from llm.llm_manager import LLMManager  # ★ LLMRouter ではなく LLMManager を依存先にする


@dataclass
class MemoryRecord:
    """
    単一の「長期記憶メモ」レコード。

    - id        : 連番 ID（メモごとに一意）
    - round_id  : そのメモが追加された会話ラウンド番号
    - summary   : 要約された記憶内容（人間が読んで分かる文章）
    - source    : "dialogue" など、由来の簡易ラベル
    """
    id: int
    round_id: int
    summary: str
    source: str = "dialogue"


class MemoryAI:
    """
    会話ログから「長期的に重要そうな記憶」を抽出・蓄積し、
    次のターンで使うためのコンテキスト文字列を生成するクラス。

    - LLM による重要事項抽出（ごく簡易な要約）
    - Streamlit の session_state を使ったメモリストア
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        persona_id: str = "default",
        model_name: str = "gpt51",
        max_records: int = 100,
        temperature: float = 0.2,
        max_tokens: int = 400,
    ) -> None:
        """
        Args:
            llm_manager : LLMManager インスタンス（AnswerTalker から渡される）
            persona_id  : この記憶が紐づくペルソナ ID（例: "floria_ja"）
            model_name  : 抽出に使うモデル名（"gpt51" / "gpt4o" / "hermes" 等）
            max_records : 保存する最大メモ数（超えたら古いものから削除）
        """
        self.llm_manager = llm_manager
        self.persona_id = persona_id
        self.model_name = model_name
        self.max_records = max_records
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        # session_state 上のキー
        self._store_key = f"memory_store_{self.persona_id}"
        self._next_id_key = f"memory_next_id_{self.persona_id}"

        self._ensure_state_initialized()

    # ============================
    # 内部: セッション状態の初期化
    # ============================
    def _ensure_state_initialized(self) -> None:
        if self._store_key not in st.session_state:
            st.session_state[self._store_key] = []
        if self._next_id_key not in st.session_state:
            st.session_state[self._next_id_key] = 1

    # ============================
    # 内部: ストア操作
    # ============================
    def _get_store(self) -> List[Dict[str, Any]]:
        self._ensure_state_initialized()
        store = st.session_state.get(self._store_key)
        if not isinstance(store, list):
            store = []
            st.session_state[self._store_key] = store
        return store

    def _set_store(self, store: List[Dict[str, Any]]) -> None:
        st.session_state[self._store_key] = store

    def _next_id(self) -> int:
        self._ensure_state_initialized()
        nid = int(st.session_state.get(self._next_id_key, 1))
        st.session_state[self._next_id_key] = nid + 1
        return nid

    # ============================
    # 内部: LLM 呼び出しヘルパ
    # ============================
    def _call_llm_for_summary(self, transcript: str) -> Optional[str]:
        """
        会話ログ（ユーザー発言＋最終返答）から、
        「長期的に重要そうな事実」を 1〜3 文に要約してもらう。

        重要な点が無い場合は "NONE" と返すよう依頼する。
        """
        if not transcript.strip():
            return None

        system_prompt = (
            "あなたは会話ログから「長期的に重要になりそうな事実」だけを抽出するアシスタントです。"
            "以下のログを読み、キャラクターにとって今後の会話で参照した方がよさそうな情報だけを、"
            "日本語で 1〜3 文程度に要約してください。"
            "もし特に重要な情報が無いと判断した場合は、必ず 'NONE' とだけ出力してください。"
        )
        user_prompt = (
            "【会話ログ】\n"
            f"{transcript}\n\n"
            "【出力形式】\n"
            "- 重要な事実がある場合: 1〜3文の日本語の要約\n"
            "- 重要な事実がない場合: 'NONE'\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            # ★ LLMManager に「モデル名で投げる」
            raw = self.llm_manager.call_model(
                name=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception:
            return None

        # call_model は LLMRouter 経由で (reply_text, usage) を返す設計
        if isinstance(raw, tuple) and raw:
            reply_text = raw[0]
        else:
            reply_text = raw

        if not isinstance(reply_text, str):
            return None

        text = reply_text.strip()
        if not text or text.upper() == "NONE":
            return None

        return text

    # ============================
    # 公開: コンテキスト生成
    # ============================
    def build_memory_context(
        self,
        user_query: str,
        max_items: int = 5,
    ) -> str:
        """
        現在のメモリストアから、system メッセージとして挿入する
        コンテキスト文字列を組み立てて返す。

        現状はシンプルに「新しい順に max_items 件」を利用する。
        将来的に user_query に応じた関連度フィルタを加えてもよい。
        """
        store = self._get_store()
        if not store:
            return ""

        records: List[MemoryRecord] = [MemoryRecord(**r) for r in store]

        records_sorted = sorted(records, key=lambda r: r.id, reverse=True)
        picked = records_sorted[:max_items]
        picked = list(reversed(picked))

        lines = []
        lines.append("【長期記憶メモ（要約）】")
        lines.append(
            "これはこれまでの会話から抽出された、"
            "キャラクターにとって長期的に重要と思われる出来事や感情のメモです。"
        )
        for rec in picked:
            lines.append(f"- {rec.summary}")

        return "\n".join(lines)

    # ============================
    # 公開: ターン終了時の記憶更新
    # ============================
    def update_from_turn(
        self,
        messages: List[Dict[str, Any]],
        final_reply: str,
        round_id: int,
    ) -> Dict[str, Any]:
        """
        1ラウンド分の会話終了後に呼ばれ、
        「このラウンドから何か覚えるべきことがあるか」を判定し、
        あればメモリストアに追加する。
        """
        try:
            chunks: List[str] = []
            for m in messages:
                role = m.get("role")
                content = str(m.get("content", "")).strip()
                if not content:
                    continue
                if role == "user":
                    chunks.append(f"[USER] {content}")
                elif role == "assistant":
                    chunks.append(f"[ASSISTANT] {content}")
                elif role == "system":
                    continue

            if final_reply:
                chunks.append(f"[ASSISTANT_FINAL] {final_reply.strip()}")

            transcript = "\n".join(chunks)

            summary = self._call_llm_for_summary(transcript)

            if not summary:
                store = self._get_store()
                return {
                    "status": "skip",
                    "added": 0,
                    "total": len(store),
                    "error": None,
                }

            store = self._get_store()
            rec = MemoryRecord(
                id=self._next_id(),
                round_id=int(round_id),
                summary=summary,
                source="dialogue",
            )
            store.append(asdict(rec))

            if len(store) > self.max_records:
                overflow = len(store) - self.max_records
                store = store[overflow:]

            self._set_store(store)

            return {
                "status": "ok",
                "added": 1,
                "total": len(store),
                "error": None,
            }

        except Exception as e:
            store = self._get_store()
            return {
                "status": "error",
                "added": 0,
                "total": len(store),
                "error": str(e),
            }
