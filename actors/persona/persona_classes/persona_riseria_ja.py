# actors/persona/persona_classes/persona_riseria_ja.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from actors.emotion_levels import AffectionLevel, affection_to_level


class Persona:
    """
    リセリア・ダ・シルヴァ用 Persona。
    actors/persona/persona_datas/elf_riseria_da_silva_ja.json を読み込み、
    {PLAYER_NAME} を差し込みつつ、好感度レベルごとのヒント生成も担当する。

    主な役割:
      - system_prompt のベース文字列を返す
      - affection_levels (low/mid/high/extreme) を JSON から読み出す
      - 好感度スコア or レベルから [感情状態ヒント] テキストを組み立てる
    """

    def __init__(self, player_name: str = "アツシ") -> None:
        self.player_name = player_name
        data = self._load_json()

        # JSON の生データ
        self.raw: Dict[str, Any] = data

        self.id: str = data.get("id", "elf_riseria_da_silva_ja")
        self.display_name: str = data.get("display_name", "リセリア・ダ・シルヴァ")

        # system_prompt 内の {PLAYER_NAME} を差し替え
        sp = data.get("system_prompt", "") or ""
        self.system_prompt: str = sp.replace("{PLAYER_NAME}", player_name)

    # =====================================================
    # JSON 読み込み
    # =====================================================
    def _load_json(self) -> Dict[str, Any]:
        """
        actors/persona/persona_datas/elf_riseria_da_silva_ja.json をロードする。
        """
        here = Path(__file__).resolve().parent
        # ……/actors/persona/persona_classes/ → ……/actors/persona/persona_datas/
        json_path = here.parent / "persona_datas" / "elf_riseria_da_silva_ja.json"

        text = json_path.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("persona JSON must be an object")
        return data

    # =====================================================
    # 公開インターフェース（Actor / ModelsAI から利用想定）
    # =====================================================
    def get_system_prompt(self) -> str:
        """
        ベースとなる system_prompt（好感度レベル非依存）を返す。
        """
        return self.system_prompt

    # -------------------------
    # affection_levels 周り
    # -------------------------
    def get_affection_level_block(self, level: AffectionLevel) -> Dict[str, Any]:
        """
        JSON の affection_levels から、指定レベル(low/mid/high/extreme)の
        設定ブロックを返す。見つからなければ {}。
        """
        raw_levels = self.raw.get("affection_levels") or {}
        if not isinstance(raw_levels, dict):
            return {}
        block = raw_levels.get(level) or {}
        return block if isinstance(block, dict) else {}

    def build_affection_hint_from_level(self, level: AffectionLevel) -> str:
        """
        指定レベル（low/mid/high/extreme）に対応する
        [感情状態ヒント] テキストを組み立てる。

        ModelsAI / AnswerTalker が system_prompt の末尾に
        そのまま追記できるフォーマットを想定。
        """
        block = self.get_affection_level_block(level)
        if not block:
            return ""  # レベル未定義なら何も出さない

        desc = str(block.get("description", "") or "").strip()
        hints = block.get("speech_hints") or []
        if not isinstance(hints, list):
            hints = []

        lines: list[str] = [
            "[感情状態ヒント]",
            f"- 現在の好感度レベル: {level}",
        ]
        if desc:
            # {PLAYER_NAME} をここでも差し替え
            lines.append(
                f"- 状態の説明: {desc.replace('{PLAYER_NAME}', self.player_name)}"
            )

        if hints:
            lines.append("- 振る舞いの具体例:")
            for h in hints:
                h_str = str(h or "")
                if "{PLAYER_NAME}" in h_str:
                    h_str = h_str.replace("{PLAYER_NAME}", self.player_name)
                lines.append(f"  - {h_str}")

        return "\n".join(lines)

    def build_affection_hint_from_score(self, score: float) -> str:
        """
        0.0〜1.0 の好感度スコアからレベルを判定し、
        そのレベルに対応する [感情状態ヒント] を返すヘルパ。

        将来的に：
          - doki_power を加味した affection_with_doki
        をそのまま渡せるようにしておく想定。
        """
        level = affection_to_level(score)
        return self.build_affection_hint_from_level(level)
