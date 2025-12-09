# actors/narrator/narrator_ai/narrator_ai.py など
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
    - label: ボタン名（UI用）
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
    LLM 呼び出し自体は NarratorManager に委譲する。

    partner_role / partner_name を受け取り、
    フローリア固定ではなく「相手キャラクター汎用」で動作する。
    """

    def __init__(
        self,
        manager: NarratorManager,
        *,
        partner_role: str = "floria",
        partner_name: str = "フローリア",
    ) -> None:
        self.manager = manager
        self.partner_role = partner_role
        self.partner_name = partner_name

    # ============================================================
    # 内部ヘルパ：SceneAI から一括でワールド情報を取る
    # ============================================================
    def _get_scene_snapshot(self) -> Dict[str, Any]:
        """
        SceneAI 経由で world_state を取得し、
        Narrator 用の簡易ビューを作る。

        互換性のため、戻り値には
        - "floria_location"  も残しておくが、実体は partner の位置。
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

        # Dokipower / SceneEmotion / その他から crowd_mode を拾えるだけ拾う
        crowd_mode = (
            party.get("crowd_mode")
            or ws.get("crowd_mode")
            or ws.get("dokipower", {}).get("crowd_mode")
            or ws.get("scene_emotion", {}).get("crowd_mode")
            or "auto"
        )

        weather = ws.get("weather", "clear")

        player_loc = locs.get("player", "プレイヤーの部屋")
        partner_loc = locs.get(self.partner_role, player_loc)

        party_mode = party.get("mode")
        if not party_mode:
            # 念のため自己判定
            party_mode = "both" if player_loc == partner_loc else "alone"

        time_slot = t.get("slot", "morning")
        time_str = t.get("time_str", "07:30")

        return {
            "world_state": ws,
            "player_location": player_loc,
            # 互換用キー（中身は partner の位置）
            "floria_location": partner_loc,
            # 新しい汎用キー
            "partner_location": partner_loc,
            "partner_role": self.partner_role,
            "partner_name": self.partner_name,
            "party_mode": party_mode,       # "both" or "alone"
            "crowd_mode": crowd_mode,       # "auto" / "alone" / "with_others"
            "time_slot": time_slot,
            "time_str": time_str,
            "weather": weather,
        }

    # ------------------------------------------------------------
    # crowd_mode に応じた説明文（Round0 用）
    # ------------------------------------------------------------
    def _build_crowd_instruction_for_system(
        self,
        crowd_mode: str,
        location_label: str,
    ) -> str:
        """
        system_prompt に埋め込む「周囲の人の扱い」の指示文を、
        DokipowerControl の crowd_mode に応じて切り替える。
        """
        if crowd_mode == "alone":
            # モブ完全禁止
            return (
                f"- 周囲には他の生徒はいません。{location_label}は "
                "プレイヤーと相手キャラクターの二人きりの静かな空間として描写してください。"
            )
        elif crowd_mode == "with_others":
            # モブ必ず出す（ただし背景）
            return (
                f"- 周囲には他の生徒たちもいますが、あくまで背景として軽く描写する程度にとどめ、"
                "視点と感情の主役はプレイヤーと相手キャラクターになるようにしてください。"
            )
        else:
            # auto（従来仕様）
            return (
                "- 周囲の人の有無は、学校の施設として不自然でない範囲で、"
                "少数のモブがいるか、静かな二人きりかのどちらかとして描写してください。"
            )

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
        Round0 用プロンプトを、パーティ状態と crowd_mode に応じて切り替える。
        - party_mode == "both"  : プレイヤー＋相手キャラが同じ場所にいる導入
        - party_mode == "alone" : プレイヤー一人きりの導入（相手キャラは本文に出さない）
        """
        snap = self._get_scene_snapshot()
        ws = snap["world_state"]
        player_loc = snap["player_location"]
        partner_loc = snap["partner_location"]
        party_mode = snap["party_mode"]
        time_slot = snap["time_slot"]
        time_str = snap["time_str"]
        weather = snap["weather"]
        partner_name = self.partner_name

        # DokipowerControl 側から crowd_mode が乗ってくる想定
        crowd_mode = snap.get("crowd_mode", "auto")
        # 念のため player_profile / floria_state 側も見る
        player_profile = player_profile or {}
        floria_state = floria_state or {}
        crowd_mode = (
            player_profile.get("crowd_mode")
            or floria_state.get("crowd_mode")
            or crowd_mode
        )

        partner_mood = floria_state.get("mood", "少し緊張している")

        # スロット名をざっくりした日本語にマッピング（ヒント用）
        slot_label_map = {
            "morning": "朝",
            "lunch": "昼",
            "after_school": "放課後",
            "night": "夜",
        }
        time_of_day_jp = slot_label_map.get(time_slot, time_slot)

        # 場所ラベル（system の説明用）
        location_label = player_loc

        # crowd_mode に応じた一行の指示文
        crowd_line = self._build_crowd_instruction_for_system(
            crowd_mode=crowd_mode,
            location_label=location_label,
        )

        if party_mode == "alone":
            # ===== プレイヤー一人きりバージョン =====
            system = f"""
あなたは会話RPG「Lyra」の中立的なナレーターです。

- プレイヤーキャラクターが一人きりでいる状況の「ラウンド0」の導入文を書きます。
- プレイヤーは、これから誰かと出会ったり、話しかけたりする前の状態です。
- 二人称または三人称の「地の文」で、2〜4文程度。
- プレイヤーや他キャラクターのセリフは一切書かない（台詞は禁止）。
- このシーンの相手キャラクター（例: フローリアやリセリア）はこの場にいません。
  本文中にその相手キャラクターの名前や存在を出してはいけません。
{crowd_line}
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
- 周囲の状況: crowd_mode={crowd_mode}

[要件]
上記の状況にふさわしい導入ナレーションを、2〜4文の地の文だけで書いてください。
- プレイヤーの一人きりの空気感や、これから何かが起こりそうな予感を描写してください。
- 相手キャラクター（{partner_name}）の名前や存在には一切触れないでください。
- JSON や説明文は書かず、物語の本文だけを書きます。
""".strip()

        else:
            # ===== プレイヤー＋相手キャラクター同伴バージョン =====
            system = f"""
あなたは会話RPG「Lyra」の中立的なナレーターです。

- プレイヤーと「相手キャラクター」（例: フローリアやリセリア）が、
  まだ一言も発言していない「ラウンド0」の導入文を書きます。
- 二人称または三人称の「地の文」で、2〜4文程度。
- プレイヤーや相手キャラクターのセリフは一切書かない（台詞は禁止）。
- 二人の距離感や空気、これから会話が始まりそうな雰囲気をさりげなく示してください。
{crowd_line}
- 最後の1文には、プレイヤーが何か話しかけたくなるような、ささやかなフックを入れてください。
- 文体は落ち着いた日本語ライトノベル調。過度なギャグやメタ発言は禁止。
""".strip()

            user = f"""
[ワールド状態（内部表現）]
world_state: {ws}

[シーン情報（プレイヤー＋相手キャラクター）]
- 現在地: {player_loc}
- 時刻帯: {time_of_day_jp}（slot={time_slot}, time={time_str}）
- 天候: {weather}
- 相手キャラクター名: {partner_name}
- {partner_name} の雰囲気・感情: {partner_mood}
- プレイヤーと {partner_name} は、今この場所で一緒にいますが、まだ一言も会話を交わしていません。
- 周囲の状況: crowd_mode={crowd_mode}

[要件]
上記のシーンにふさわしい導入ナレーションを、2〜4文の地の文だけで書いてください。
- プレイヤーと {partner_name} の距離感や、これから会話が始まりそうな空気感を中心に描写してください。
- プレイヤーや {partner_name} のセリフは書かないでください（地の文のみ）。
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
        world_state 引数は互換のため受け取るが、
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
    # ここから下は前と同じ（_build_action_messages / _refine /
    # make_wait_choice / make_look_person_choice / make_scan_area_choice /
    # make_special_title_and_choice）
    # ============================================================
    # ……（ここは前バージョンをそのまま使ってOK）
