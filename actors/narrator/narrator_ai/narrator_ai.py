# actors/narrator/narrator_ai/narrator_ai.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import streamlit as st

from actors.narrator.narrator_manager import NarratorManager
from actors.scene_ai import SceneAI
from actors.init_ai import InitAI
from .types import NarrationLine, NarrationChoice

# LLMManager を直接同期できるようにする（manager側に無い場合の保険）
from llm.llm_manager import LLMManager


@dataclass
class NarratorAI:
    """
    プロンプト設計＋最終テキストの「意味づけ」担当。
    LLM 呼び出し自体は NarratorManager に丸投げする。

    partner_role / partner_name を受け取り、
    フローリア固定ではなく「相手キャラクター汎用」で動作する。
    """

    manager: NarratorManager
    partner_role: str = "floria"
    partner_name: str = "フローリア"

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
    # 内部ヘルパ：InitAI の初期化保証（現行 init_ai.py に合わせる）
    # ============================================================
    def _ensure_initialized(self) -> None:
        """
        起動直後（Round0）でも player_name / manual_controls / world_state が未定義にならないように、
        InitAI をここでも保険として呼ぶ。

        重要: 旧コードの InitAI.apply(ctx, ...) は現行 init_ai.py に存在しない前提なので使わない。
        """
        try:
            # st.session_state を Mapping として渡す（InitAI 側が dict write を try してくれている）
            InitAI.ensure_minimum(state=st.session_state, persona=None, snapshot=None)
        except Exception:
            # ここで落として会話全体を壊したくないので握りつぶす
            pass

    # ============================================================
    # 内部ヘルパ：AIManager 設定（enabled/priority）を Narrator 側へ同期
    # ============================================================
    def _sync_ai_manager_settings(self) -> None:
        """
        st.session_state["ai_manager"] の設定を読み、
        NarratorManager が使う LLMManager に enabled_models を反映する。

        - enabled_models: 提出AIを On/Off
        - priority: （使えるなら）優先順位も manager 側へ渡す
        """
        ai_state = st.session_state.get("ai_manager")
        if not isinstance(ai_state, dict):
            return

        enabled = ai_state.get("enabled_models")
        priority = ai_state.get("priority")

        # ---- enabled_models を LLMManager に反映 ----
        if isinstance(enabled, dict):
            # 1) manager が llm_manager を持っているならそこへ
            mgr_llm = getattr(self.manager, "llm_manager", None)

            if mgr_llm is not None and hasattr(mgr_llm, "set_enabled_models"):
                try:
                    mgr_llm.set_enabled_models(enabled)
                except Exception:
                    pass
            else:
                # 2) 無ければ default に反映（最低限「どれを呼ぶか」を揃える）
                try:
                    LLMManager.get_or_create("default").set_enabled_models(enabled)
                except Exception:
                    pass

        # ---- priority を（可能なら）manager側へ渡す ----
        if isinstance(priority, list) and priority:
            # manager 側に優先順位のsetterがあるなら使う（無ければ session_state に残すだけ）
            if hasattr(self.manager, "set_priority"):
                try:
                    self.manager.set_priority([str(x) for x in priority])
                except Exception:
                    pass

            # デバッグ/互換のため、セッションにも置いておく（他の層が参照しやすい）
            st.session_state["narrator_priority"] = [str(x) for x in priority]

    # ============================================================
    # 内部ヘルパ：SceneAI から一括でワールド情報を取る
    # ============================================================
    def _get_scene_snapshot(self) -> Dict[str, Any]:
        """
        SceneAI 経由で world_state を取得し、Narrator 用の簡易ビューを作る。
        """
        self._ensure_initialized()
        self._sync_ai_manager_settings()

        scene_ai = SceneAI(state=st.session_state)

        ws = scene_ai.get_world_state() or {}
        if not isinstance(ws, dict):
            ws = {}

        player_name = str(ws.get("player_name") or st.session_state.get("player_name") or "アツシ").strip() or "アツシ"

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

        default_home = f"{player_name}の部屋"

        player_loc = locs.get("player") or default_home
        if player_loc == "プレイヤーの部屋":
            player_loc = default_home

        partner_loc = locs.get(self.partner_role, None)
        if partner_loc == "プレイヤーの部屋":
            partner_loc = default_home

        # ========= others_present 判定 =========
        others_present: Optional[bool] = None

        flags = ws.get("flags")
        if isinstance(flags, dict) and "others_present" in flags:
            v = flags.get("others_present")
            if isinstance(v, bool):
                others_present = v

        if others_present is None and "others_present" in ws:
            v = ws.get("others_present")
            if isinstance(v, bool):
                others_present = v

        if others_present is None:
            manual_ws = st.session_state.get("world_state_manual_controls") or {}
            v = manual_ws.get("others_present")
            if isinstance(v, bool):
                others_present = v

        if others_present is None:
            manual_emo = st.session_state.get("emotion_manual_controls") or {}
            v = manual_emo.get("others_present")
            if isinstance(v, bool):
                others_present = v
            else:
                env = manual_emo.get("environment")
                if env == "with_others":
                    others_present = True
                elif env == "alone":
                    others_present = False

        if others_present is None:
            others_present = False

        # ========= party_mode =========
        party_mode = party.get("mode")
        if party_mode not in ("both", "alone"):
            if partner_loc is None:
                party_mode = "alone"
            else:
                party_mode = "both" if player_loc == partner_loc else "alone"

        if partner_loc is None:
            partner_loc = "（この場にはいない）"

        # ========= interaction_mode_hint =========
        interaction_mode: str = "auto"
        manual_ws = st.session_state.get("world_state_manual_controls") or {}
        hint = manual_ws.get("interaction_mode_hint")
        if isinstance(hint, str) and hint in (
            "auto",
            "auto_with_others",
            "pair_private",
            "pair_public",
            "solo",
            "solo_with_others",
        ):
            interaction_mode = hint
        else:
            manual_emo = st.session_state.get("emotion_manual_controls") or {}
            hint2 = manual_emo.get("interaction_mode_hint")
            if isinstance(hint2, str) and hint2 in (
                "auto",
                "auto_with_others",
                "pair_private",
                "pair_public",
                "solo",
                "solo_with_others",
            ):
                interaction_mode = hint2

        # --- 手動ヒントを確定 ---
        if interaction_mode in ("pair_private", "pair_public", "solo", "solo_with_others"):
            if interaction_mode == "pair_private":
                party_mode = "both"
                others_present = False
            elif interaction_mode == "pair_public":
                party_mode = "both"
                others_present = True
            elif interaction_mode == "solo":
                party_mode = "alone"
                others_present = False
            elif interaction_mode == "solo_with_others":
                party_mode = "alone"
                others_present = True
        else:
            if interaction_mode == "auto_with_others":
                others_present = True

            if party_mode == "both":
                interaction_mode = "pair_public" if others_present else "pair_private"
            else:
                interaction_mode = "solo_with_others" if others_present else "solo"

        time_slot = t.get("slot", "morning")
        time_str = t.get("time_str", "07:30")

        return {
            "world_state": ws,
            "player_name": player_name,
            "player_location": player_loc,
            "floria_location": partner_loc,   # 互換用キー
            "partner_location": partner_loc,
            "partner_role": self.partner_role,
            "partner_name": self.partner_name,
            "party_mode": party_mode,              # "both" / "alone"
            "others_present": bool(others_present),
            "interaction_mode": interaction_mode,  # pair_private / pair_public / solo / solo_with_others
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
        snap = self._get_scene_snapshot()
        ws = snap["world_state"]
        player_name: str = snap.get("player_name", "アツシ") or "アツシ"
        player_loc = snap["player_location"]
        interaction_mode: str = snap.get("interaction_mode", "pair_private")
        others_present: bool = bool(snap.get("others_present", False))
        time_slot = snap["time_slot"]
        time_str = snap["time_str"]
        weather = snap["weather"]
        partner_name = self.partner_name

        player_profile = player_profile or {}
        partner_state = floria_state or {}
        partner_mood = partner_state.get("mood", "少し緊張している")

        slot_label_map = {
            "morning": "朝",
            "lunch": "昼",
            "after_school": "放課後",
            "night": "夜",
        }
        time_of_day_jp = slot_label_map.get(time_slot, time_slot)

        if interaction_mode in ("solo", "solo_with_others"):
            system = """
あなたは会話RPG「Lyra」の中立的なナレーターです。

- プレイヤーキャラクターが一人きりでいる状況の「ラウンド0」の導入文を書きます。
- 二人称または三人称の「地の文」で、2〜4文程度。
- プレイヤーや他キャラクターのセリフは一切書かない（台詞は禁止）。
- このシーンの相手キャラクターはこの場にいません。本文で相手キャラクターの名前や存在を出してはいけません。
- 周囲に他の生徒たちの気配があるかどうかは、指定フラグに従うこと。
- 最後の1文に、プレイヤーが行動したくなる「ささやかなフック」を入れる。
- 文体は落ち着いた日本語ライトノベル調。メタ発言は禁止。
""".strip()

            env_label = (
                "周囲には他の生徒たちの気配もある（あなたは一人だが、完全な静寂ではない）"
                if interaction_mode == "solo_with_others"
                else "周囲には他の生徒たちの気配はなく、完全に一人きり"
            )

            user = f"""
[ワールド状態（内部表現）]
world_state: {ws}

[プレイヤー情報（必須）]
- プレイヤー名: {player_name}
- 現在地: {player_loc}
- 時刻帯: {time_of_day_jp}（slot={time_slot}, time={time_str}）
- 天候: {weather}
- 周囲の状況: {env_label}

[要件]
上記の状況にふさわしい導入ナレーションを、2〜4文の地の文だけで書いてください。
- 本文中でプレイヤー名「{player_name}」を **最低1回は必ず** 使ってください。
- 相手キャラクター（{partner_name}）の名前や存在には一切触れないでください。
- JSON や説明文は書かず、物語の本文だけを書きます。
""".strip()

        else:
            system = """
あなたは会話RPG「Lyra」の中立的なナレーターです。

- プレイヤーと相手キャラクターが、まだ一言も発言していない「ラウンド0」の導入文を書きます。
- 二人称または三人称の「地の文」で、2〜4文程度。
- プレイヤーや相手キャラクターのセリフは一切書かない（台詞は禁止）。
- 周囲の人の有無は「二人きり」か「他の生徒たちもいる」かとして、指定フラグに必ず従うこと。
- 二人の距離感や空気、これから会話が始まりそうな雰囲気をさりげなく示す。
- 最後の1文に、プレイヤーが話しかけたくなる「ささやかなフック」を入れる。
- 文体は落ち着いた日本語ライトノベル調。メタ発言は禁止。
""".strip()

            env_label = (
                "同じ場所には二人以外にも、少数の他の生徒たちがいます"
                if (interaction_mode == "pair_public" or others_present)
                else "この場所にいるのはプレイヤーと相手キャラクターの二人きりです"
            )

            user = f"""
[ワールド状態（内部表現）]
world_state: {ws}

[プレイヤー情報（必須）]
- プレイヤー名: {player_name}

[シーン情報（プレイヤー＋相手キャラクター）]
- 現在地: {player_loc}
- 時刻帯: {time_of_day_jp}（slot={time_slot}, time={time_str}）
- 天候: {weather}
- 相手キャラクター名: {partner_name}
- {partner_name} の雰囲気・感情: {partner_mood}
- プレイヤーと {partner_name} は、今この場所で一緒にいますが、まだ一言も会話を交わしていません。
- 周囲の状況: {env_label}

[要件]
上記のシーンにふさわしい導入ナレーションを、2〜4文の地の文だけで書いてください。
- 本文中でプレイヤー名「{player_name}」を **最低1回は必ず** 使ってください。
- 「他の生徒たちもいる」場合は、その存在が自然に伝わる描写を1箇所以上入れてください。
- 「二人きり」の場合は、他の生徒たちの存在を一切出さず、静かな二人きりの空気感として描写してください。
- セリフは書かないでください（地の文のみ）。
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
            meta={"scene_snapshot": snap},
        )

    # ============================================================
    # 行動整形（wait / look_person / scan_area / special）
    # ============================================================
    def _build_action_messages(self, intent_text: str, label: str) -> List[Dict[str, str]]:
        snap = self._get_scene_snapshot()
        ws = snap["world_state"]
        player_name: str = snap.get("player_name", "アツシ") or "アツシ"
        player_loc = snap["player_location"]
        partner_loc = snap["partner_location"]
        party_mode = snap["party_mode"]
        time_slot = snap["time_slot"]
        time_str = snap["time_str"]
        weather = snap["weather"]
        partner_name = self.partner_name
        others_present: bool = bool(snap.get("others_present", False))
        interaction_mode: str = snap.get("interaction_mode", "pair_private")

        if interaction_mode == "solo_with_others":
            env_label = "あなた以外にも学院生たちの気配がある（相手キャラクターはこの場にいない）"
        elif interaction_mode == "solo":
            env_label = "この場にいるのはあなただけで、静かな空気が満ちている"
        elif interaction_mode == "pair_public":
            env_label = "あなたと相手キャラクターのほかに、周囲には何人かの学院生たちがいる"
        else:
            env_label = "あなたと相手キャラクターの二人きりの空間になっている"

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

[プレイヤー情報（必須）]
- プレイヤー名: {player_name}

[シーン要約]
- パーティ状態: {party_mode}（"both"=プレイヤーと相手キャラが一緒, "alone"=プレイヤーだけ）
- プレイヤーの現在地: {player_loc}
- 相手キャラクター({partner_name})の現在地: {partner_loc}
- 時刻帯: {time_slot}（time={time_str}）
- 天候: {weather}
- シーン種別: {interaction_mode}
- 周囲の状況: {env_label}

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
        from .make_wait_choice import build_wait_choice
        return build_wait_choice(self, world_state=world_state, floria_state=floria_state)

    def make_look_person_choice(
        self,
        actor_name: str = "フローリア",
        actor_id: Optional[str] = None,
        world_state: Dict[str, Any] | None = None,
        floria_state: Dict[str, Any] | None = None,
    ) -> NarrationChoice:
        from .make_scan_area_choice import build_look_person_choice
        return build_look_person_choice(
            self,
            actor_name=actor_name,
            actor_id=actor_id,
            world_state=world_state,
            floria_state=floria_state,
        )

    def make_scan_area_choice(
        self,
        location_name: str = "この場",
        world_state: Dict[str, Any] | None = None,
        floria_state: Dict[str, Any] | None = None,
    ) -> NarrationChoice:
        from .make_scan_area_choice import build_scan_area_choice
        return build_scan_area_choice(
            self,
            location_name=location_name,
            world_state=world_state,
            floria_state=floria_state,
        )

    def make_special_title_and_choice(
        self,
        special_id: str,
        world_state: Dict[str, Any] | None = None,
        floria_state: Dict[str, Any] | None = None,
    ) -> tuple[str, NarrationChoice]:
        from .make_special_title_and_choice import build_special_title_and_choice
        return build_special_title_and_choice(
            self,
            special_id=special_id,
            world_state=world_state,
            floria_state=floria_state,
        )
