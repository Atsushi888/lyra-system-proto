from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict
import json
import os

import streamlit as st


@dataclass
class PersonaAI:
    """
    Persona 用 JSON を読み込み、
    内部にそのまま保持しつつ、必要に応じて「全部さらす」クラス。

    - persona_id: "floria_ja" など
    - base_dir  : JSON を置いているディレクトリ
                  （現状: actors/persona_datas/<persona_id>.json）
    """

    persona_id: str = "floria_ja"
    base_dir: str = "actors/persona_datas"

    # 最後に読み込んだ JSON 生データ
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def json_path(self) -> str:
        return os.path.join(self.base_dir, f"{self.persona_id}.json")

    # ----------------------------------
    # world_state ヘルパ
    # ----------------------------------
    def get_world_state(self) -> Dict[str, Any]:
        """
        現在の world_state を返すヘルパ。
        Persona 情報に「今どこで何時か」を埋め込むために使う。
        """
        loc = st.session_state.get("scene_location", "通学路")
        slot = st.session_state.get("scene_time_slot")
        tstr = st.session_state.get("scene_time_str")
        return {
            "location": loc,
            "time_slot": slot,
            "time_str": tstr,
        }

    # ----------------------------------
    # 1) JSON 読み込み
    # ----------------------------------
    def load_from_json(self) -> None:
        """
        JSON ファイルを読み込んで self.data に格納する。
        ファイルが無い / 壊れている場合は、data を {} にしておく。
        """
        path = self.json_path
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                self.data = raw
            else:
                # 想定外形式なら、とりあえず空にする
                self.data = {}
        except FileNotFoundError:
            # 初回などファイルが無いケース
            self.data = {}
        except Exception as e:
            # 何かあったらログ用にメッセージだけ残して空に
            print(f"[PersonaAI] failed to load JSON: {path} ({e})")
            self.data = {}

    # ----------------------------------
    # 2) 内部データを「全部さらす」
    # ----------------------------------
    def get_all(self, reload: bool = False) -> Dict[str, Any]:
        """
        Persona JSON の中身をそのまま返す。

        - reload=True のときは毎回 JSON を再読込してから返す。
        - reload=False なら、前回 load_from_json 済みの内容を返す。
        """
        if reload or not self.data:
            self.load_from_json()

        result = dict(self.data)
        # world_state を一緒に埋め込む
        result["world_state"] = self.get_world_state()
        return result

    # ----------------------------------
    # 3) 将来用：LLM用 system_prompt を組み立てるフック
    # ----------------------------------
    def build_system_prompt_for_llm(self) -> str:
        """
        将来的に「JSON の中身から system_prompt を再構成したい」とき用のフック。
        いまは単純に data["system_prompt"] を返す。
        """
        if not self.data:
            self.load_from_json()
        return str(self.data.get("system_prompt", ""))
