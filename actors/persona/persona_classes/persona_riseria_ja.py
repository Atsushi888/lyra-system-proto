# actors/persona/persona_classes/persona_riseria_ja.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class Persona:
    """
    リセリア・ダ・シルヴァ用 Persona。

    - /actors/persona/persona_datas/elf_riseria_da_silva_ja.json を読み込み
    - {PLAYER_NAME} プレースホルダを差し替えた system_prompt を提供
    - Actor / AnswerTalker から呼ばれる build_messages() を実装
    """

    JSON_NAME = "elf_riseria_da_silva_ja.json"

    def __init__(self, player_name: str = "アツシ") -> None:
        self.player_name = player_name

        data = self._load_json()

        # JSON の生データ
        self.raw: Dict[str, Any] = data

        # 基本プロファイル
        self.id: str = data.get("id", "elf_riseria_da_silva_ja")
        self.display_name: str = data.get("display_name", "リセリア・ダ・シルヴァ")
        self.short_name: str = data.get("short_name", "リセ")

        # system_prompt 内の {PLAYER_NAME} を差し替え
        base_sp = data.get("system_prompt", "")
        self.system_prompt: str = base_sp.replace("{PLAYER_NAME}", player_name)

    # --------------------------------------------------
    # JSON ロード
    # --------------------------------------------------
    def _load_json(self) -> Dict[str, Any]:
        """
        /actors/persona/persona_datas/elf_riseria_da_silva_ja.json
        から Persona 定義を読み込む。
        """
        here = Path(__file__).resolve().parent
        # .../actors/persona/persona_classes/ → persona_datas/ へ
        json_path = here.parent / "persona_datas" / self.JSON_NAME

        text = json_path.read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        return {}

    # --------------------------------------------------
    # 公開 API
    # --------------------------------------------------
    def get_system_prompt(self) -> str:
        """Actor / AnswerTalker などから参照される想定のヘルパ。"""
        return self.system_prompt

    def build_messages(
        self,
        user_text: str,
        conversation_log: Optional[List[Dict[str, str]]] = None,
        world_state: Optional[Dict[str, Any]] = None,
        affection_hint: Optional[str] = None,
        extra_system_hint: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Actor → AnswerTalker に渡すための messages を構築する。

        いまのところシンプルに
          - リセリアの system_prompt
          - （必要なら）好感度ヒントや追加システム指示
          - プレイヤー発話（user_text）
        だけを詰める。

        conversation_log / world_state は将来拡張用の受け皿として受け取るだけで、
        現段階では使わない（互換性確保のために引数だけ受けて捨てる）。
        """
        system_parts: List[str] = [self.system_prompt]

        if extra_system_hint:
            extra_system_hint = extra_system_hint.strip()
            if extra_system_hint:
                system_parts.append(extra_system_hint)

        if affection_hint:
            affection_hint = affection_hint.strip()
            if affection_hint:
                system_parts.append(affection_hint)

        system_text = "\n\n".join(system_parts)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        return messages
