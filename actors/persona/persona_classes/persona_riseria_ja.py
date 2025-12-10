# actors/persona/persona_classes/persona_riseria_ja.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

from actors.persona.persona_base.persona_base import PersonaBase


class Persona(PersonaBase):
    """
    リセリア・ダ・シルヴァ用 Persona。

    - PersonaBase を継承し、JSON_NAME だけ指定する
    - system_prompt / emotion_profiles 周りはすべて PersonaBase 側の実装を利用
    """

    JSON_NAME = "elf_riseria_da_silva_ja.json"

    def __init__(self, player_name: str = "アツシ") -> None:
        super().__init__(player_name=player_name)
        # 必要ならここでリセリア固有の追加初期化を行う
        import streamlit as st
        st.write("=== DEBUG: Persona(Riseria) loaded ===")
        st.write(f"id: {self.id!r}")
        st.write(f"display_name: {self.display_name!r}")
        st.write(f"short_name: {self.short_name!r}")
        st.write(f"system_prompt_prefix: {self.system_prompt[:120]!r}")
