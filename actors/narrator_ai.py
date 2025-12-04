# actors/narrator_ai.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Dict, Any, Optional, List

import streamlit as st

from actors.narrator.narrator_manager import NarratorManager
from actors.scene_ai import SceneAI


ChoiceKind = Literal["round0", "look_person", "scan_area", "wait", "special"]


@dataclass
class NarrationLine:
    """
    ナレーション1本分。
    Round0 など「地の文だけ返してほしい」用途。
    """
    text: str
    kind: ChoiceKind = "round0"
    meta: Dict[str, Any] | None = None


@dataclass
class NarrationChoice:
    """
    プレイヤーが「救済」として選ぶ行動テキスト。
    - label: ボタン名（UI用）※今は固定でOK
    - speak_text: 実際にログに出す地の文
    """
    kind: ChoiceKind
    label: str
    speak_text: str
    target_id: Optional[str] = None
    special_id: Optional[str] = None
    meta: Dict[str, Any] | None = None


class NarratorAI:
    """
    プロンプト設計＋最終テキストの「意味づけ」担当。
    LLM 呼び出し自体は NarratorManager に丸投げする。

    ※世界情報（world_state）は SceneAI から直接取得する。
      CouncilManager などから渡される world_state 引数は
      後方互換のために受け取るが、基本的には無視する方針。
    """

    def __init__(self, manager: NarratorManager) -> None:
        self.manager = manager

    # ============================================================
    # 内部ヘルパ：SceneAI から一括でワールド情報を取る
    # ============================================================
    def _get_scene_snapshot(self) -> Dict[str, Any]:
        """
        SceneAI 経由で world_state を取得し、
        Narrator 用の簡易ビューを作る。
        """
        scene_ai = SceneAI(state=st.session_state)
        ws = scene_ai.get_world_state()

        locs = ws.get("locations") or {}
        if not isinstance(locs, dict):
            locs = {}

        t = ws.get("time") or {}
        if not isinstance(t, dict):
            t = {}

        party = ws.get("party") or {}
        if not isinstance(party, dict):
            party = {}

        weather = ws.get("weather", "clear")

        player_loc = locs.get("player", "プレイヤーの部屋")
        floria_loc = locs.get("floria", player_loc)

        party_mode = party.get("mode")
        if not party_mode:
            # 念のため自己判定
            party_mode = "both" if player_loc == floria_loc else "alone"

        time_slot = t.get("slot", "morning")
        time_str = t.get("time_str", "07:30")

        return {
            "world_state": ws,
            "player_location": player_loc,
            "floria_location": floria_loc,
            "party_mode": party_mode,  # "both" or "alone"
            "time_slot": time_slot,
            "time_str": time_str,
            "weather": weather,
        }

    # ============================================================
    # Round0 導入ナレーション
    # ============================================================
    def _build_round0_messages(
        self,
        world_state: Dict[str, Any] | None = None,
        player_profile: Dict[str, Any] | None = None,
        floria_state: Dict[str, Any] | None = None,
    ) -> List[Dict[str, str]]:
        """
        Round0 用プロンプトを、パーティ状態に応じて 2 パターンで切り替える。
        - party_mode == "both"  : プレイヤー＋フローリアが同じ場所にいる導入
        - party_mode == "alone" : プレイヤー一人きりの導入（フローリアは本文に出さない）
        """
        snap = self._get_scene_snapshot()
        ws = snap["world_state"]
        player_loc = snap["player_location"]
        floria_loc = snap["floria_location"]
        party_mode = snap["party_mode"]
        time_slot = snap["time_slot"]
        time_str = snap["time_str"]
        weather = snap["weather"]

        player_profile = player_profile or {}
        floria_state = floria_state or {}
        floria_mood = floria_state.get("mood", "少し緊張している")

        # スロット名をざっくりした日本語にマッピング（ヒント用）
        slot_label_map = {
            "morning": "朝",
            "lunch": "昼",
            "after_school": "放課後",
            "night": "夜",
        }
        time_of_day_jp = slot_label_map.get(time_slot, time_slot)

        if party_mode == "alone":
            # ===== プレイヤー一人きりバージョン =====
            system = """
あなたは会話RPG「Lyra」の中立的なナレーターです。

- プレイヤーキャラクターが一人きりでいる状況の「ラウンド0」の導入文を書きます。
- プレイヤーは、これから誰かと出会ったり、話しかけたりする前の状態です。
- 二人称または三人称の「地の文」で、2〜4文程度。
- プレイヤーや他キャラクターのセリフは一切書かない（台詞は禁止）。
- 現時点ではフローリアはこの場にいません。本文中にフローリアの名前や存在を出してはいけません。
- 最後の1文には、プレイヤーがこれから誰かに会ったり、行動を起こしたくなるような、ささやかなフックを入れてください。
- 文体は落ち着いた日本語ライトノベル調。過度なギャグやメタ発言は禁止。
""".strip()

            user = f"""
[ワールド状態（内部表現）]
world_state: {ws}

[シーン情報（プレイヤーのみ）]
- 現在地: {player_loc}
- 時刻帯: {time_of_day_jp}（slot={time_slot}, time={time_str}）
- 天候: {weather}
- 現在この場にいるのはプレイヤーだけです。フローリアは別の場所にいます。

[要件]
上記の状況にふさわしい導入ナレーションを、2〜4文の地の文だけで書いてください。
- プレイヤーの一人きりの空気感や、これから何かが起こりそうな予感を描写してください。
- フローリアの名前や存在には一切触れないでください。
- JSON や説明文は書かず、物語の本文だけを書きます。
""".strip()

        else:
            # ===== プレイヤー＋フローリア同伴バージョン =====
            system = """
あなたは会話RPG「Lyra」の中立的なナレーターです。

- プレイヤーとフローリアが、まだ一言も発言していない「ラウンド0」の導入文を書きます。
- 二人称または三人称の「地の文」で、2〜4文程度。
- プレイヤーやフローリアのセリフは一切書かない（台詞は禁止）。
- 二人の距離感や空気、これから会話が始まりそうな雰囲気をさりげなく示してください。
- 最後の1文には、プレイヤーが何か話しかけたくなるような、ささやかなフックを入れてください。
- 文体は落ち着いた日本語ライトノベル調。過度なギャグやメタ発言は禁止。
""".strip()

            user = f"""
[ワールド状態（内部表現）]
world_state: {ws}

[シーン情報（プレイヤー＋フローリア）]
- 現在地: {player_loc}
- 時刻帯: {time_of_day_jp}（slot={time_slot}, time={time_str}）
- 天候: {weather}
- フローリアの雰囲気・感情: {floria_mood}
- プレイヤーとフローリアは、今この場所で一緒にいますが、まだ一言も会話を交わしていません。

[要件]
上記のシーンにふさわしい導入ナレーションを、2〜4文の地の文だけで書いてください。
- プレイヤーとフローリアの距離感や、これから会話が始まりそうな空気感を中心に描写してください。
- プレイヤーやフローリアのセリフは書かないでください（地の文のみ）。
- JSON や説明文は書かず、物語の本文だけを書きます。
""".strip()

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def generate_round0_opening(
        self,
        world_state: Dict[str, Any],
        player_profile: Dict[str, Any],
        floria_state: Dict[str, Any],
    ) -> NarrationLine:
        """
        ※ world_state 引数は互換のため受け取るが、
           実際には SceneAI 側の world_state を参照する。
        """
        messages = self._build_round0_messages(
            world_state=world_state,
            player_profile=player_profile,
            floria_state=floria_state,
        )
        text = self.manager.run_task(
            task_type="round0",
            label="Round0 opening",
            messages=messages,
            mode_current="narrator_round0",
        )
        snap = self._get_scene_snapshot()
        return NarrationLine(
            text=text,
            kind="round0",
            meta={
                "scene_snapshot": snap,
            },
        )

    # ============================================================
    # 行動整形（wait / look_person / scan_area / special）
    # ============================================================
    def _build_action_messages(
        self,
        intent_text: str,
        label: str,
    ) -> List[Dict[str, str]]:
        """
        共通の「行動→地の文」Refiner 用プロンプト。
        world_state は毎回 SceneAI から取得する。
        """
        snap = self._get_scene_snapshot()
        ws = snap["world_state"]
        player_loc = snap["player_location"]
        floria_loc = snap["floria_location"]
        party_mode = snap["party_mode"]
        time_slot = snap["time_slot"]
        time_str = snap["time_str"]
        weather = snap["weather"]

        system = """
あなたは会話RPG「Lyra」のナレーション補助AIです。

ユーザーが指定した「行動の意図」を、ライトノベル風の自然な日本語 1〜2 文に整えます。

- 行動の意図（何をしたいか）は絶対に変えない。
- 性的・暴力的な強度を勝手に盛らない。暗示レベルに留める。
- フローリアなどキャラクターのセリフは書かない。プレイヤー側の動きだけを書く。
- 1〜2文以内に収める。
""".strip()

        user = f"""
[ワールド状態（内部表現）]
world_state: {ws}

[シーン要約]
- パーティ状態: {party_mode}（"both"=プレイヤーとフローリアが一緒, "alone"=プレイヤーだけ）
- プレイヤーの現在地: {player_loc}
- フローリアの現在地: {floria_loc}
- 時刻帯: {time_slot}（time={time_str}）
- 天候: {weather}

[行動の意図]
「{intent_text}」

上記の意図をそのまま維持したまま、ライトノベル風の地の文 1〜2 文に整えてください。
出力は本文のみとし、余計な説明やラベルは書かないこと。
""".strip()

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _refine(self, intent_text: str, label: str) -> str:
        messages = self._build_action_messages(intent_text, label=label)
        return self.manager.run_task(
            task_type="action",
            label=label,
            messages=messages,
            mode_current="narrator_action",
        )

    # ============================================================
    # 公開API: 救済用 Choice
    # ============================================================
    def make_wait_choice(
        self,
        world_state: Dict[str, Any] | None = None,
        floria_state: Dict[str, Any] | None = None,
    ) -> NarrationChoice:
        """
        「何もしない」。
        プレイヤー一人（party_mode == "alone"）のときは、
        ここでも「誰もいないのでここでじっとしているしかない」旨を返す。
        """
        snap = self._get_scene_snapshot()
        party_mode = snap["party_mode"]

        if party_mode == "alone":
            speak = (
                "この場には他に誰の気配もない。"
                "あなたはしばし足を止め、静かな空気の中で次の出会いを待つことにした。"
            )
        else:
            intent = "何もせず、しばらく黙って様子を見る"
            speak = self._refine(intent_text=intent, label="wait")

        return NarrationChoice(
            kind="wait",
            label="何もしない",
            speak_text=speak,
        )

    def make_look_person_choice(
        self,
        actor_name: str = "フローリア",
        actor_id: Optional[str] = None,
        world_state: Dict[str, Any] | None = None,
        floria_state: Dict[str, Any] | None = None,
    ) -> NarrationChoice:
        """
        「相手の様子を伺う」。
        プレイヤー一人（party_mode == "alone"）のときは、
        そもそも相手がいない旨だけを返す。
        """
        snap = self._get_scene_snapshot()
        party_mode = snap["party_mode"]

        if party_mode == "alone":
            speak = (
                "周囲を見回してみるが、この場にあなた以外の人影はない。"
                "様子をうかがう相手が現れるのを、もう少し待つしかなさそうだ。"
            )
        else:
            intent = f"{actor_name}の様子をうかがう"
            speak = self._refine(intent_text=intent, label="look_person")

        return NarrationChoice(
            kind="look_person",
            label=f"{actor_name}の様子をうかがう",
            speak_text=speak,
            target_id=actor_id,
            meta={"actor_name": actor_name},
        )

    def make_scan_area_choice(
        self,
        location_name: str = "この場",
        world_state: Dict[str, Any] | None = None,
        floria_state: Dict[str, Any] | None = None,
    ) -> NarrationChoice:
        """
        「周囲の様子を見る」。
        プレイヤー一人でも有効だが、
        party_mode==alone の場合は「誰もいない静かな情景」を中心に描く。
        """
        snap = self._get_scene_snapshot()
        party_mode = snap["party_mode"]
        loc = snap["player_location"] or location_name

        if party_mode == "alone":
            # 一人のときは、静かなロケーション描写に寄せる
            intent = f"{loc}の周囲の静かな様子を改めて見回す"
        else:
            intent = f"{loc}の周囲の様子を見回す"

        speak = self._refine(intent_text=intent, label="scan_area")

        return NarrationChoice(
            kind="scan_area",
            label="周囲の様子を見る",
            speak_text=speak,
            meta={"location": loc},
        )

    def make_special_title_and_choice(
        self,
        special_id: str,
        world_state: Dict[str, Any] | None = None,
        floria_state: Dict[str, Any] | None = None,
    ) -> tuple[str, NarrationChoice]:
        """
        スペシャルアクションは、プレイヤー単独でも実行可能。
        world_state は SceneAI から取得したものを前提に Refine する。
        """
        if special_id == "touch_pillar":
            title = "古い石柱に手を触れる"
            intent = "目の前の古い石柱に手を伸ばして触れる"
        elif special_id == "pray_to_moon":
            title = "月へ祈りを捧げる"
            intent = "静かに目を閉じて月へ祈りを捧げる"
        else:
            title = "特別な行動を取る"
            intent = "胸の内の衝動に従い、特別な行動をひとつ取る"

        speak = self._refine(intent_text=intent, label=f"special:{special_id}")

        choice = NarrationChoice(
            kind="special",
            label=title,
            speak_text=speak,
            special_id=special_id,
        )
        return title, choice
