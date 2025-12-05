# actors/persona/persona_classes/persona_riseria_ja.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class Persona:
    """
    リセリア・ダ・シルヴァ用 Persona。
    actors/persona/persona_datas/elf_riseria_da_silva_ja.json を読み込み、
    system_prompt 内の {PLAYER_NAME} を差し替える。
    """

    def __init__(self, player_name: str = "アツシ") -> None:
        self.player_name = player_name

        data = self._load_json()

        # JSON の中身
        self.raw: Dict[str, Any] = data

        self.id: str = data.get("id", "elf_riseria_da_silva_ja")
        self.display_name: str = data.get("display_name", "リセリア・ダ・シルヴァ")

        # system_prompt 内の {PLAYER_NAME} を置換
        sp = data.get("system_prompt", "")
        self.system_prompt: str = sp.replace("{PLAYER_NAME}", player_name)

    # -------------------------------------------------------
    # JSON ローダ
    # -------------------------------------------------------
    def _load_json(self) -> Dict[str, Any]:
        """
        現在のファイル位置:
            actors/persona/persona_classes/persona_riseria_ja.py

        読みたい JSON の位置:
            actors/persona/persona_datas/elf_riseria_da_silva_ja.json
        """

        here = Path(__file__).resolve().parent               # persona_classes/
        persona_dir = here.parent / "persona_datas"          # persona/persona_datas/
        json_path = persona_dir / "elf_riseria_da_silva_ja.json"

        text = json_path.read_text(encoding="utf-8")
        return json.loads(text)

    # -------------------------------------------------------
    # 必要インターフェース
    # -------------------------------------------------------
    def get_system_prompt(self) -> str:
        return self.system_prompt
