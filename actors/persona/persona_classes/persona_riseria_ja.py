from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from actors.scene_ai import SceneAI


class Persona:
    """
    リセリア・ダ・シルヴァ用 Persona。
    actors/persona/persona_datas/elf_riseria_da_silva_ja.json を読み込み、
    system_prompt 内の {PLAYER_NAME} を差し替える。

    Actor クラスからは Floria 用 Persona と同等に扱えるよう、
    build_messages(conversation_log) を実装する。
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

        self.starter_hint: str = str(data.get("starter_hint", ""))
        self.style_hint: str = str(data.get("style_hint", ""))

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
    # Actor から呼ばれるインターフェース
    # -------------------------------------------------------
    def get_system_prompt(self) -> str:
        return self.system_prompt

    def _build_world_context_system(self) -> str:
        """
        SceneAI から現在のワールド情報を軽く読み取り、
        「いまどこで、どんな状況か」をリセリア用に要約した system メッセージを返す。
        """
        try:
            scene_ai = SceneAI(state=st.session_state)
            ws = scene_ai.get_world_state()
        except Exception:
            ws = {}

        locs = ws.get("locations") or {}
        t = ws.get("time") or {}

        player_loc = str(locs.get("player", ""))
        time_slot = str(t.get("slot", ""))
        time_str = str(t.get("time_str", ""))

        # ざっくりラベル化
        slot_label_map = {
            "morning": "朝",
            "lunch": "昼",
            "after_school": "放課後",
            "night": "夜",
        }
        time_of_day_jp = slot_label_map.get(time_slot, time_slot or "不明な時間帯")

        # ここで「プレイヤーの部屋＋朝」のときは
        # それとなく“お泊まり明け”っぽい空気を意識させる。
        base = (
            f"【状況メモ（リセリア用）】\n"
            f"- 現在の場所: {player_loc or '不明な場所'}\n"
            f"- 時刻帯: {time_of_day_jp}（{time_str or '時刻不明'}）\n"
            f"- リセリアは今、このシーンでプレイヤーと同じ場所にいる。\n"
        )

        if "部屋" in player_loc and time_slot == "morning":
            # プレイヤーの部屋の朝 ⇒ 少し気まずさ・照れを強調
            base += (
                "- 朝の静かな空気の中、ベッドや部屋の雰囲気から、"
                "昨夜ふたりがかなり親しい時間を過ごしたかもしれないことを、"
                "リセリアは少し意識している。\n"
                "- ただし、あからさまに性行為を描写したりはせず、"
                "『一緒に夜を過ごしたかもしれない』程度の、"
                "ふわっとした記憶と恥ずかしさとして扱うこと。\n"
                "- 彼女の振る舞いには、場所の親密さや距離の近さに対する"
                "ささやかな照れや意識の揺れをにじませてよい。\n"
            )
        else:
            base += (
                "- リセリアは現在の場所や時間帯にふさわしい振る舞いを心がけるが、"
                "過剰に状況説明を喋りすぎないこと。\n"
            )

        return base

    def build_messages(self, conversation_log: Any) -> List[Dict[str, str]]:
        """
        Actor.speak(...) から呼び出されることを想定したメッセージ構築。

        引数 conversation_log には、通常 CouncilManager が保持している
        List[Dict[str, str]] 形式の会話ログが渡される：

            {"role": "player" | "riseria" | "narrator" | ...,
             "content": "<br>区切りのテキスト"}

        これを OpenAI / LLM 用の chat messages に変換する。
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        # 文体ヒント
        if self.style_hint:
            messages.append({
                "role": "system",
                "content": f"[Style Hint]\n{self.style_hint}",
            })

        # ★ ワールド状況をリセリア用に要約したメモを system として追加
        world_context = self._build_world_context_system()
        if world_context:
            messages.append({"role": "system", "content": world_context})

        log: List[Dict[str, str]] = []
        if isinstance(conversation_log, list):
            log = conversation_log

        for entry in log:
            role = entry.get("role", "")
            text = (entry.get("content") or "").replace("<br>", "\n")

            if role == "player":
                messages.append({"role": "user", "content": text})
            elif role in ("riseria", "floria"):
                messages.append({"role": "assistant", "content": text})
            elif role == "narrator":
                messages.append(
                    {"role": "system", "content": f"[ナレーション]\n{text}"}
                )
            else:
                messages.append({"role": "system", "content": text})

        return messages
