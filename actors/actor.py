# actors/actor.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from conversation_engine import LLMConversation  # 既存
# Persona は厳密な型がまだ変わるかもなので Any で受ける
# 必要なら: from personas.persona_floria_ja import Persona など


ActorKind = Literal["player", "npc", "gm"]


@dataclass
class Actor:
    """
    Lyra-System 上の「登場人物」を表す最小単位。

    - persona: その Actor の人格・システムプロンプト
    - kind: player / npc / gm などの役割
    - history: この Actor 視点での会話ログ（role, content のペア）
    - temperature / max_tokens: デフォルト値（あとでペルソナ別に上書き可）

    現段階では「1 Actor = 1 LLMConversation」で、
    将来的にここに multi-AI / DeliberationBundle / LLMRouter を差し込む。
    """

    id: str
    display_name: str
    persona: Any  # 将来 Persona 型に差し替え
    kind: ActorKind = "npc"

    # LLM 呼び出しパラメータ（仮置き）
    temperature: float = 0.7
    max_tokens: int = 800

    # この Actor が見ている会話履歴
    history: List[Dict[str, str]] = field(default_factory=list)

    # 任意のメタ情報（シナリオ用フラグなど）
    meta: Dict[str, Any] = field(default_factory=dict)

    # ===== 内部ユーティリティ =====

    def _build_system_prompt(self) -> str:
        """
        Persona から system_prompt を組み立てる。
        いまの persona 実装に合わせて、必要ならここを書き換える。
        """
        # persona が build_system_prompt() を持っている前提で呼び出し、
        # 無ければ fallback として str(persona) を使う。
        if hasattr(self.persona, "build_system_prompt"):
            return self.persona.build_system_prompt()
        if hasattr(self.persona, "system_prompt"):
            return str(self.persona.system_prompt)
        return str(self.persona)

    def _build_style_hint(self) -> str:
        """
        文体ヒント。persona 側にあれば拾う。
        無ければ LLMConversation の default を使わせる。
        """
        return getattr(self.persona, "style_hint", "")

    def _create_engine(self) -> LLMConversation:
        """
        この Actor 専用の LLMConversation を1回ぶん生成する。
        将来はここを差し替えるだけで multi-AI 審議に移行できる。
        """
        return LLMConversation(
            system_prompt=self._build_system_prompt(),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            style_hint=self._build_style_hint(),
        )

    # ===== ログ操作 =====

    def add_user_message(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.history.append({"role": "assistant", "content": content})

    def add_system_message(self, content: str) -> None:
        self.history.append({"role": "system", "content": content})

    # ===== メイン API：Actor にしゃべらせる =====

    def speak(
        self,
        user_text: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Actor に「ユーザーからこう言われた」という前提で返答させる。

        - history に user → assistant を積む
        - LLMConversation.generate_reply() を内部で呼ぶ
        - 将来はここを DeliberationBundle / LLMRouter に差し替える
        """
        # 1) ユーザー発話を履歴に追加
        self.add_user_message(user_text)

        # 2) LLM 会話エンジンを生成して呼び出し
        engine = self._create_engine()
        reply_text, llm_meta = engine.generate_reply(self.history)

        # 3) 返答を履歴に積む
        self.add_assistant_message(reply_text)

        # 4) 結果を返す（ゲーム側は reply_text だけ使ってもOK）
        return reply_text, llm_meta

    # ===== デバッグ / シリアライズ補助 =====

    def history_as_text(self) -> str:
        """デバッグ用：ログをざっくりテキスト化。"""
        lines: List[str] = []
        for m in self.history:
            role = m.get("role", "?")
            content = m.get("content", "")
            lines.append(f"[{role}] {content}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """
        セーブデータ化用の素直な dict。
        persona は ID だけ残して、実体は別レジストリ管理…なども後で可能。
        """
        return {
            "id": self.id,
            "display_name": self.display_name,
            "kind": self.kind,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "history": list(self.history),
            "meta": dict(self.meta),
            # persona はとりあえずクラス名だけ吐く
            "persona_class": self.persona.__class__.__name__,
        }
