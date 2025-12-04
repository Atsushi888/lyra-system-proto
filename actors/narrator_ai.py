from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Dict, Any, Optional, List

from actors.narrator.narrator_manager import NarratorManager


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
    """

    def __init__(self, manager: NarratorManager) -> None:
        self.manager = manager

    # ============================================================
    # Round0 導入ナレーション
    # ============================================================
    def _analyze_party_state(
        self,
        world_state: Dict[str, Any],
        floria_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        world_state から「プレイヤーが一人か / フローリアと同じ場所にいるか」を判定する。

        できるだけ堅く判定するため、
        - party_state フラグ
        - locations.player / locations.floria
        の両方を見る。
        """
        party_raw = str(world_state.get("party_state", "")).strip().lower()

        # SceneAI 側で locations を持っている想定:
        #   locations = {"player": "...", "floria": "..."}
        locations = world_state.get("locations") or {}
        player_loc = (
            world_state.get("player_location")
            or world_state.get("location_name")
            or locations.get("player")
            or "不明な場所"
        )
        floria_loc = (
            world_state.get("floria_location")
            or locations.get("floria")
            or floria_state.get("location")
            or ""
        )

        # 「同じ場所にいる」とみなす条件
        together_by_flag = party_raw in ("with_floria", "together", "pair")
        together_by_loc = bool(floria_loc) and (player_loc == floria_loc)
        together = together_by_flag or together_by_loc

        if party_raw == "alone":
            # 明示的に alone が立っている場合は最優先で 1人扱い
            together = False

        return {
            "party_state": party_raw or None,
            "player_loc": player_loc,
            "floria_loc": floria_loc or None,
            "is_together": together,
        }

    def _build_round0_messages(
        self,
        world_state: Dict[str, Any],
        player_profile: Dict[str, Any],
        floria_state: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """
        Round0 用プロンプト。
        - プレイヤー単独
        - フローリア同席
        でプロンプトを切り替える。
        """
        info = self._analyze_party_state(world_state, floria_state)
        location_name = world_state.get("location_name", info["player_loc"])
        time_of_day = world_state.get("time_of_day", "不明")
        weather = world_state.get("weather", "不明")

        # --- 分岐: 一緒かどうか ---
        if info["is_together"]:
            # ▼ プレイヤー + フローリアが「同じ場所」にいる場合
            system = """
あなたは会話RPG「Lyra」の中立的なナレーターです。

- プレイヤーとフローリアが、まだ一言も発言していない「ラウンド0」の導入文を書きます。
- 二人称または三人称の「地の文」で、2〜4文程度。
- プレイヤーやフローリアのセリフは一切書かない（台詞は禁止）。
- 最後の1文には、プレイヤーが何か話しかけたくなるような、ささやかなフックを入れてください。
- 文体は落ち着いた日本語ライトノベル調。過度なギャグやメタ発言は禁止。
""".strip()

            user = f"""
[シーン情報]
- 場所: {location_name}
- 時刻帯: {time_of_day}
- 天候: {weather}
- パーティ状態: プレイヤーとフローリアが同じ場所にいる
- フローリアの雰囲気・感情: {floria_state.get("mood", "不明")}

[要件]
上記のシーンにふさわしい導入ナレーションを、2〜4文の地の文だけで書いてください。
プレイヤーとフローリアはすでに同じ場所におり、物理的に近くに存在していますが、
セリフは一切書かず、状況描写と心情の雰囲気だけを描いてください。
JSON や説明文は書かず、物語の本文だけを書きます。
""".strip()

        else:
            # ▼ プレイヤーが一人（またはフローリアが別の場所にいる）場合
            system = """
あなたは会話RPG「Lyra」の中立的なナレーターです。

- プレイヤーが「一人で」いるシーン、もしくは
  フローリアが別の場所にいて現在の場にはいないシーンの導入文を書きます。
- 二人称または三人称の「地の文」で、2〜4文程度。
- フローリアはこの場には登場しません。
  彼女の姿や具体的な行動を、現在のシーンの中で描写してはいけません。
  （遠くにいる・あとで会う予定だといった言及までは許可）
- プレイヤーやフローリアのセリフは一切書かない（台詞は禁止）。
- 最後の1文には、プレイヤーが何か行動したくなるような、ささやかなフックを入れてください。
- 文体は落ち着いた日本語ライトノベル調。過度なギャグやメタ発言は禁止。
""".strip()

            floria_loc = info["floria_loc"] or "別の場所"
            party_desc = "プレイヤーは一人でいる（フローリアは現在この場にはいない）"

            user = f"""
[シーン情報]
- 場所: {location_name}
- 時刻帯: {time_of_day}
- 天候: {weather}
- パーティ状態: {party_desc}
- フローリアの現在位置（参考情報）: {floria_loc}

[要件]
上記のシーンにふさわしい導入ナレーションを、2〜4文の地の文だけで書いてください。
プレイヤーは現在ひとりきりであり、この場にフローリアは存在しません。
フローリアが「ここにいる」「隣に立っている」「隣で話しかけてくる」など、
この場に登場しているような描写は絶対に書かないでください。
必要であれば、彼女が離れた場所にいる・これから会いに行く予定であるといった
軽い言及だけに留めてください。

JSON や説明文は書かず、物語の本文だけを書きます。
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
        messages = self._build_round0_messages(world_state, player_profile, floria_state)
        text = self.manager.run_task(
            task_type="round0",
            label="Round0 opening",
            messages=messages,
            mode_current="narrator_round0",
        )

        # デバッグ用に party 判定結果も meta に入れておく
        party_info = self._analyze_party_state(world_state, floria_state)

        meta = {
            "world_state": world_state,
            "player_profile": player_profile,
            "floria_state": floria_state,
            "party_info": party_info,
        }

        return NarrationLine(
            text=text,
            kind="round0",
            meta=meta,
        )

    # ============================================================
    # 行動整形（wait / look_person / scan_area / special）
    # ============================================================
    def _build_action_messages(
        self,
        intent_text: str,
        world_state: Dict[str, Any] | None = None,
        floria_state: Dict[str, Any] | None = None,
    ) -> List[Dict[str, str]]:
        world_state = world_state or {}
        floria_state = floria_state or {}

        system = """
あなたは会話RPG「Lyra」のナレーション補助AIです。

ユーザーが指定した「行動の意図」を、ライトノベル風の自然な日本語 1〜2 文に整えます。

- 行動の意図（何をしたいか）は絶対に変えない。
- 性的・暴力的な強度を勝手に盛らない。暗示レベルに留める。
- フローリアなどキャラクターのセリフは書かない。プレイヤー側の動きだけを書く。
- 1〜2文以内に収める。
""".strip()

        user = f"""
[シーン情報（ヒント）]
world_state: {world_state}
floria_state: {floria_state}

[行動の意図]
「{intent_text}」

上記の意図をそのまま維持したまま、ライトノベル風の地の文 1〜2 文に整えてください。
出力は本文のみとし、余計な説明やラベルは書かないこと。
""".strip()

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _refine(
        self,
        intent_text: str,
        label: str,
        world_state: Dict[str, Any] | None = None,
        floria_state: Dict[str, Any] | None = None,
    ) -> str:
        messages = self._build_action_messages(intent_text, world_state, floria_state)
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
        intent = "何もせず、しばらく黙って様子を見る"
        speak = self._refine(
            intent,
            label="wait",
            world_state=world_state,
            floria_state=floria_state,
        )
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
        intent = f"{actor_name}の様子をうかがう"
        speak = self._refine(
            intent,
            label="look_person",
            world_state=world_state,
            floria_state=floria_state,
        )
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
        intent = f"{location_name}の周囲の様子を見回す"
        speak = self._refine(
            intent,
            label="scan_area",
            world_state=world_state,
            floria_state=floria_state,
        )
        return NarrationChoice(
            kind="scan_area",
            label="周囲の様子を見る",
            speak_text=speak,
            meta={"location": location_name},
        )

    def make_special_title_and_choice(
        self,
        special_id: str,
        world_state: Dict[str, Any] | None = None,
        floria_state: Dict[str, Any] | None = None,
    ) -> tuple[str, NarrationChoice]:
        if special_id == "touch_pillar":
            title = "古い石柱に手を触れる"
            intent = "目の前の古い石柱に手を伸ばして触れる"
        elif special_id == "pray_to_moon":
            title = "月へ祈りを捧げる"
            intent = "静かに目を閉じて月へ祈りを捧げる"
        else:
            title = "特別な行動を取る"
            intent = "胸の内の衝動に従い、特別な行動をひとつ取る"

        speak = self._refine(
            intent,
            label=f"special:{special_id}",
            world_state=world_state,
            floria_state=floria_state,
        )

        choice = NarrationChoice(
            kind="special",
            label=title,
            speak_text=speak,
            special_id=special_id,
        )
        return title, choice
