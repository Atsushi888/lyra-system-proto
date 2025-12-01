# actors/narrator_ai.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Dict, Any, Optional

from llm.llm_manager import LLMManager
from llm.llm_manager_factory import get_llm_manager


ChoiceKind = Literal["round0", "look_person", "scan_area", "wait", "special"]


@dataclass
class NarrationLine:
    """
    ナレーション1本分。
    Round0 など「地の文だけ返してほしい」用途に使う。
    """
    text: str
    kind: ChoiceKind = "round0"
    meta: Dict[str, Any] | None = None


@dataclass
class NarrationChoice:
    """
    プレイヤーが「救済措置」として選べる行動。
    - kind: look_person / scan_area / wait / special
    - label: UIボタンに表示する短いテキスト
    - speak_text: ログに残すプレイヤーの行動テキスト
    """
    kind: ChoiceKind
    label: str
    speak_text: str
    target_id: Optional[str] = None
    special_id: Optional[str] = None
    meta: Dict[str, Any] | None = None


class NarratorAI:
    """
    Lyra 用ナレーター（雛形版）。

    - Round0 用の導入地の文
    - 救済行動（wait / look_person / scan_area / special）のテキスト生成

    いったんはハードコード中心で、あとから LLM 連携に差し替え可能な構造。
    """

    def __init__(
        self,
        llm_manager: Optional[LLMManager] = None,
        persona_id: str = "narrator",
    ) -> None:
        # いまは使っていないが、将来 LLM 連携で使う余地を残す
        self.llm_manager = llm_manager or get_llm_manager(persona_id)

    # ============ Round0 ============

    def generate_round0_opening(
        self,
        world_state: Dict[str, Any] | None = None,
        player_profile: Dict[str, Any] | None = None,
        floria_state: Dict[str, Any] | None = None,
    ) -> NarrationLine:
        """
        Round0 用の導入ナレーション。
        ひとまずハードコード版。TODO: あとで LLM プロンプトに差し替え。
        """
        text = (
            "夜風が静かに路地を撫でていた。"
            "青白い月明かりの下、あなたとフローリアは並んで立っている。"
            "言葉を探すような沈黙が、二人のあいだに落ちていた。"
        )
        return NarrationLine(text=text, kind="round0", meta={"hook": "round0_opening"})

    # ============ 救済行動：何もしない ============

    def make_wait_choice(self) -> NarrationChoice:
        return NarrationChoice(
            kind="wait",
            label="何もしない",
            speak_text="しばし沈黙を保つ。",
        )

    # ============ 救済行動：相手の様子を伺う ============

    def make_look_person_choice(
        self,
        actor_name: str = "フローリア",
        actor_id: Optional[str] = None,
    ) -> NarrationChoice:
        """
        「相手の様子を伺う」救済行動。
        まずはシンプルな固定文。あとから LLM へ差し替え可能。
        """
        label = f"{actor_name}の様子をうかがう"
        speak = f"そっと{actor_name}の横顔に視線を向ける。"
        return NarrationChoice(
            kind="look_person",
            label=label,
            speak_text=speak,
            target_id=actor_id,
            meta={"actor_name": actor_name},
        )

    # ============ 救済行動：周りの様子を見る ============

    def make_scan_area_choice(
        self,
        location_name: str = "この場",
    ) -> NarrationChoice:
        label = "周囲の様子を見る"
        speak = "周囲の様子を静かに見回す。"
        return NarrationChoice(
            kind="scan_area",
            label=label,
            speak_text=speak,
            meta={"location": location_name},
        )

    # ============ 救済行動：スペシャル（足場だけ） ============

    def make_special_title_and_choice(
        self,
        special_id: str,
    ) -> tuple[str, NarrationChoice]:
        """
        special_id ごとにタイトルと speak_text を返す足場。
        TODO: 実際のシナリオに合わせて分岐を増やす。
        """
        if special_id == "touch_pillar":
            title = "古い石柱に手を触れる"
            speak = "目の前の古い石柱に、そっと手を伸ばして触れる。"
        elif special_id == "pray_to_moon":
            title = "月へ祈りを捧げる"
            speak = "静かに目を閉じ、月へ向けて短い祈りを捧げる。"
        else:
            title = "特別な行動を取る"
            speak = "胸の内の衝動に従い、特別な行動をひとつ選ぶ。"

        choice = NarrationChoice(
            kind="special",
            label=title,
            speak_text=speak,
            special_id=special_id,
        )
        return title, choice
