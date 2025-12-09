# actors/narrator/narrator_ai/narrator_ai.py
from __future__ import annotations

from typing import Dict, Any, List

import streamlit as st

from actors.narrator.narrator_manager import NarratorManager
from actors.scene_ai import SceneAI

from .types import ChoiceKind, NarrationLine, NarrationChoice
from .make_wait_choice import make_wait_choice_impl
from .make_scan_area_choice import make_scan_area_choice_impl
from .make_special_title_and_choice import make_special_title_and_choice_impl


class NarratorAI:
    """
    プロンプト設計＋最終テキストの「意味づけ」担当。
    LLM 呼び出し自体は NarratorManager に丸投げする。

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

        if not isinstance(ws, dict):
            ws = {}

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
            "party_mode": party_mode,  # "both" or "alone"
            "time_slot": time_slot,
            "time_str": time_str,
            "weather": weather,
        }

    # ------------------------------------------------------------
    # crowd 情報の簡易解析（Round0 用）
    # ------------------------------------------------------------
    def _analyze_crowd(self, snap: Dict[str, Any]) -> Dict[str, str]:
        """
        場所と party_mode から、モブの呼び方などを決める。
        """
        ws = snap.get("world_state") or {}
        locs = ws.get("locations") or {}
        if not isinstance(locs, dict):
            locs = {}

        player_loc = str(locs.get("player", snap.get("player_location", "")))
        party = ws.get("party") or {}
        if not isinstance(party, dict):
            party = {}

        party_mode = str(snap.get("party_mode") or "")

        # DokipowerControl が with_others を立てているならそれを優先
        with_others_flag = party.get("with_others")
        if isinstance(with_others_flag, bool):
            crowd_mode = "with_others" if with_others_flag else "alone"
        elif party_mode in ("alone", "with_others"):
            crowd_mode = party_mode
        else:
            # 場所から推定（プールはデフォルトで with_others）
            if "プール" in player_loc or "pool" in player_loc:
                crowd_mode = "with_others"
            else:
                crowd_mode = "alone"

        # 場所別のモブ呼称
        if "プール" in player_loc or "pool" in player_loc:
            mob_noun = "他の生徒たち"
        elif "教室" in player_loc or "classroom" in player_loc:
            mob_noun = "クラスメイトたち"
        else:
            mob_noun = "周囲の人々"

        return {
            "crowd_mode": crowd_mode,          # "alone" / "with_others"
            "mob_noun": mob_noun,              # 「他の生徒たち」など
            "player_location": player_loc,
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
        - crowd_mode == "with_others" : 周囲にモブがいる導入
        - crowd_mode == "alone"       : プレイヤー＋相手だけの導入
        """
        snap = self._get_scene_snapshot()
        ws = snap["world_state"]
        player_loc = snap["player_location"]
        party_mode = snap["party_mode"]
        time_slot = snap["time_slot"]
        time_str = snap["time_str"]
        weather = snap["weather"]
        partner_name = self.partner_name

        player_profile = player_profile or {}
        partner_state = floria_state or {}
        partner_mood = partner_state.get("mood", "少し緊張している")

        crowd_info = self._analyze_crowd(snap)
        crowd_mode = crowd_info["crowd_mode"]
        mob_noun = crowd_info["mob_noun"]

        # スロット名をざっくりした日本語にマッピング（ヒント用）
        slot_label_map = {
            "morning": "朝",
            "lunch": "昼",
            "after_school": "放課後",
            "night": "夜",
        }
        time_of_day_jp = slot_label_map.get(time_slot, time_slot)

        # ----------------------------------------
        # system プロンプト
        # ----------------------------------------
        base_system = """
あなたは会話RPG「Lyra」の中立的なナレーターです。

- プレイヤーと「相手キャラクター」（例: フローリアやリセリア）が、
  まだ一言も発言していない「ラウンド0」の導入文を書きます。
- 二人称または三人称の「地の文」で、2〜4文程度。
- プレイヤーや相手キャラクターのセリフは一切書かない（台詞は禁止）。
- 文体は落ち着いた日本語ライトノベル調。過度なギャグやメタ発言は禁止。
""".strip()

        if crowd_mode == "with_others":
            system = f"""{base_system}

- 周囲には {mob_noun} もいます。導入文のどこかで、{mob_noun} の存在が自然に伝わる描写を 1 箇所以上入れてください。
- ただし {mob_noun} は背景です。個別の名前を付けたり、長々と会話させたりしないでください。
- 最後の1文には、プレイヤーが何か話しかけたくなるような、ささやかなフックを入れてください。
""".strip()
        else:
            # 二人きり扱い
            system = f"""{base_system}

- 現在この場にはプレイヤーと {partner_name} 以外の人物はいません。第三者の足音や気配を匂わせる描写も入れないでください。
- 二人の距離感や空気、これから会話が始まりそうな雰囲気をさりげなく示してください。
- 最後の1文には、プレイヤーが何か話しかけたくなるような、ささやかなフックを入れてください。
""".strip()

        # ----------------------------------------
        # user プロンプト
        # ----------------------------------------
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
- 周囲の状況: crowd_mode={crowd_mode}, mob={mob_noun}

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
    # 行動整形共通（wait / look_person / scan_area / special）
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
        partner_loc = snap["partner_location"]
        party_mode = snap["party_mode"]
        time_slot = snap["time_slot"]
        time_str = snap["time_str"]
        weather = snap["weather"]
        partner_name = self.partner_name

        system = """
あなたは会話RPG「Lyra」のナレーション補助AIです。

ユーザーが指定した「行動の意図」を、ライトノベル風の自然な日本語 1〜2 文に整えます。

- 行動の意図（何をしたいか）は絶対に変えない。
- 性的・暴力的な強度を勝手に盛らない。暗示レベルに留める。
- 相手キャラクターなどのセリフは書かない。プレイヤー側の動きだけを書く。
- 1〜2文以内に収める。
""".strip()

        user = f"""
[ワールド状態（内部表現）]
world_state: {ws}

[シーン要約]
- パーティ状態: {party_mode}（"both"=プレイヤーと相手キャラが一緒, "alone"=プレイヤーだけ）
- プレイヤーの現在地: {player_loc}
- 相手キャラクター({partner_name})の現在地: {partner_loc}
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
    # ここから先は、別モジュール実装に委譲
    # ============================================================
    def make_wait_choice(
        self,
        world_state: Dict[str, Any] | None = None,
        floria_state: Dict[str, Any] | None = None,
    ) -> NarrationChoice:
        return make_wait_choice_impl(self, world_state, floria_state)

    def make_look_person_choice(
        self,
        actor_name: str = "フローリア",
        actor_id: str | None = None,
        world_state: Dict[str, Any] | None = None,
        floria_state: Dict[str, Any] | None = None,
    ) -> NarrationChoice:
        # これは元のまま（本体が短いのでここに残す）
        snap = self._get_scene_snapshot()
        party_mode = snap["party_mode"]

        # 互換のためデフォルト値は "フローリア" だが、
        # 別キャラがバインドされている場合はそちらを優先する。
        if actor_name == "フローリア" and self.partner_name != "フローリア":
            actor_name = self.partner_name

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
        return make_scan_area_choice_impl(self, location_name, world_state, floria_state)

    def make_special_title_and_choice(
        self,
        special_id: str,
        world_state: Dict[str, Any] | None = None,
        floria_state: Dict[str, Any] | None = None,
    ) -> tuple[str, NarrationChoice]:
        return make_special_title_and_choice_impl(self, special_id, world_state, floria_state)
