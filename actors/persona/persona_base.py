from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from actors.emotion_ai import EmotionResult
from actors.emotion.emotion_levels import affection_to_level
from actors.emotion.emotion_state import relationship_stage_from_level


class PersonaBase:
    """
    全 Persona の共通土台。

    - JSON_NAME で参照する JSON を決める
    - system_prompt の生成
    - messages 構築ヘルパ
    - emotion_profiles を使った好意ラベル / ドキドキガイド
    - EmotionResult / emotion_override からのヘッダ組み立て
    - relationship_level / masking_degree（ばけばけ度）の解釈
    - reply_length_mode（short/normal/long/story）の文章量ガイド
    - masking_defaults（persona JSON）＋ world_state に応じた
      「二人きりデレ解禁／人前ではばけばけ抑制」の注釈
    """

    JSON_NAME: str = ""  # 継承クラス側で上書きする想定

    def __init__(self, player_name: str = "アツシ") -> None:
        self.player_name = player_name

        data = self._load_json()
        self.raw: Dict[str, Any] = data or {}

        # 基本情報
        self.id: str = self.raw.get("id", self.JSON_NAME or "persona_base")
        self.display_name: str = self.raw.get("display_name", self.id)
        self.short_name: str = self.raw.get("short_name", self.display_name)

        # system_prompt 内の {PLAYER_NAME} を差し替え
        base_sp = self.raw.get("system_prompt", "")
        self.system_prompt: str = base_sp.replace("{PLAYER_NAME}", player_name)

    # --------------------------------------------------
    # JSON ロード
    # --------------------------------------------------
    def _load_json(self) -> Dict[str, Any]:
        """
        /actors/persona/persona_datas/{JSON_NAME} を読む。
        """
        if not self.JSON_NAME:
            return {}

        here = Path(__file__).resolve().parent
        json_path = here / "persona_datas" / self.JSON_NAME

        if not json_path.exists():
            return {}

        text = json_path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except Exception:
            return {}

        if isinstance(data, dict):
            return data
        return {}

    # --------------------------------------------------
    # system prompt / messages
    # --------------------------------------------------
    def get_system_prompt(self) -> str:
        """Actor / AnswerTalker などから参照される想定のヘルパ。"""
        return self.system_prompt

    def build_messages(
        self,
        user_text: str,
        conversation_log: Optional[List[Dict[str, str]]] = None,
        world_state: Optional[Dict[str, Any]] = None,
        affection_hint: Optional[str] = None,
        extra_system_hint: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Actor → AnswerTalker に渡すための messages を構築する共通実装。

        必要に応じてサブクラス側でオーバーライドしてもよい。
        """
        system_parts: List[str] = [self.system_prompt]

        if extra_system_hint:
            extra = extra_system_hint.strip()
            if extra:
                system_parts.append(extra)

        if affection_hint:
            ah = affection_hint.strip()
            if ah:
                system_parts.append(ah)

        system_text = "\n\n".join(system_parts)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        return messages

    # --------------------------------------------------
    # emotion_profiles（JSON）系
    # --------------------------------------------------
    def _get_emotion_profiles(self) -> Dict[str, Any]:
        """
        JSON 内の emotion_profiles セクションを返す。
        ない場合は空 dict。
        """
        return self.raw.get("emotion_profiles", {}) or {}

    def get_emotion_profile(self) -> Dict[str, Any]:
        """
        affection_gain / doki_bias など係数系。

        JSON 例:
          "emotion_profiles": {
            "profile": {
              "affection_gain": 1.2,
              "doki_bias": 1.0
            },
            ...
          }
        """
        profiles = self._get_emotion_profiles()
        prof = profiles.get("profile") or {}
        if isinstance(prof, dict):
            return prof
        return {}

    def get_affection_label(self, affection_with_doki: float) -> str:
        """
        affection_with_doki に対応する「好意の解釈」ラベルを JSON から取得。
        """
        profiles = self._get_emotion_profiles()
        labels = profiles.get("affection_labels", {}) or {}
        if not labels:
            return ""

        try:
            thresholds = sorted(
                (float(k) for k in labels.keys()),
                reverse=True,
            )
        except Exception:
            return ""

        for th in thresholds:
            if affection_with_doki >= th:
                key = f"{th:.1f}".rstrip("0").rstrip(".")
                if key in labels:
                    return labels[key]
                raw_key = str(th)
                if raw_key in labels:
                    return labels[raw_key]

        # どの閾値も満たさない場合は最小閾値のラベル
        min_th = min(thresholds)
        key = f"{min_th:.1f}".rstrip("0").rstrip(".")
        return labels.get(key, labels.get(str(min_th), ""))

    def build_emotion_control_guideline(
        self,
        *,
        affection_with_doki: float,
        doki_level: int,
        mode_current: str,
    ) -> str:
        """
        doki_level / mode に応じた「口調・距離感ガイドライン」を JSON から組み立て。
        """
        profiles = self._get_emotion_profiles()
        affection_labels = profiles.get("affection_labels", {}) or {}
        doki_levels = profiles.get("doki_levels", {}) or {}
        mode_overrides = profiles.get("mode_overrides", {}) or {}

        # 好意ラベル（あれば）
        aff_label = ""
        if affection_labels:
            aff_label = self.get_affection_label(affection_with_doki)

        # doki_level ごとの行
        doki_key = str(int(doki_level))
        doki_lines: List[str] = doki_levels.get(doki_key, []) or []

        # mode 別追加
        mode_lines: List[str] = mode_overrides.get(str(mode_current), []) or []

        lines: List[str] = []
        lines.append(f"[{self.display_name}用・口調と距離感ガイドライン]")

        if aff_label:
            lines.append(f"- 現在の好意の解釈: {aff_label}")

        if doki_lines:
            lines.extend(doki_lines)

        if mode_lines:
            lines.append("")
            lines.append("[モード別ガイドライン]")
            lines.extend(mode_lines)

        if not doki_lines and not mode_lines and not aff_label:
            lines.append(
                "※ 感情プロファイルが未設定のため、通常時とほぼ同じトーンで話してください。"
            )

        return "\n".join(lines)

    # --------------------------------------------------
    # masking_defaults（JSON）系
    # --------------------------------------------------
    def _get_masking_defaults(self) -> Dict[str, Any]:
        """
        persona JSON の masking_defaults セクションを返す。

        期待構造（例）:
          "masking_defaults": {
            "masking_level": 0.5,
            "masking_behavior": {
              "unmasked_locations": ["home", "riseria_home", "club_animal"],
              "masked_locations": ["school", "classroom", "hallway"],
              "rules": {
                "alone_bonus": "...",
                "public_softening": "...",
                "example_line": "……二人きりですね、{PLAYER_NAME}先輩…"
              }
            }
          }
        """
        raw = self.raw.get("masking_defaults") or {}
        if not isinstance(raw, dict):
            return {}

        behavior = raw.get("masking_behavior") or {}
        if not isinstance(behavior, dict):
            behavior = {}

        # 正規化して返す
        default_level = 0.0
        try:
            default_level = float(raw.get("masking_level", 0.0) or 0.0)
        except Exception:
            default_level = 0.0

        unmasked = behavior.get("unmasked_locations", []) or []
        masked = behavior.get("masked_locations", []) or []
        rules = behavior.get("rules", {}) or {}

        return {
            "default_level": default_level,
            "unmasked_locations": [str(x).lower() for x in unmasked],
            "masked_locations": [str(x).lower() for x in masked],
            "rules": rules,
        }

    # --------------------------------------------------
    # 長さモード（reply_length_mode）関連
    # --------------------------------------------------
    @staticmethod
    def _normalize_length_mode(mode: str) -> str:
        m = (mode or "auto").lower()
        if m not in ("auto", "short", "normal", "long", "story"):
            return "auto"
        return m

    def _build_length_guideline(self, length_mode: str) -> str:
        """
        reply_length_mode に応じた「文章量ガイドライン」を返す。
        auto の場合は空文字。
        """
        mode = self._normalize_length_mode(length_mode)
        if mode == "auto":
            return ""

        lines: List[str] = []
        lines.append("[文章量ガイドライン]")

        if mode == "short":
            lines.extend(
                [
                    "- 今回は短め（1〜2文程度）を目安にしてください。",
                    "- 要点だけを簡潔に伝え、余計な前置きや長い独白は避けてください。",
                ]
            )
        elif mode == "normal":
            lines.extend(
                [
                    "- 通常の会話量（3〜5文程度）を目安にしてください。",
                    "- 必要な感情描写は入れつつも、引き延ばしすぎないようにします。",
                ]
            )
        elif mode == "long":
            lines.extend(
                [
                    "- 会話中心で少し長め（5〜8文程度）を目安にしてください。",
                    "- セリフを軸にしながら、仕草や視線などの描写も適度に加えてください。",
                ]
            )
        elif mode == "story":
            lines.extend(
                [
                    "- その場の情景や雰囲気も含めたミニシーン風の返答を目安にしてください。",
                    "- セリフと地の文を組み合わせ、1つの場面として印象に残るように描写してください。",
                ]
            )

        return "\n".join(lines)

    # --------------------------------------------------
    # EmotionResult / emotion_override → system_prompt / header
    # --------------------------------------------------

    def build_emotion_based_system_prompt(
        self,
        *,
        base_system_prompt: str,
        emotion_override: Optional[Dict[str, Any]] = None,
        mode_current: str = "normal",
        length_mode: str = "auto",
    ) -> str:
        """
        emotion_override を受け取り、system_prompt に感情ヘッダ＋文章量ガイドラインを付け足したものを返す。

        主に扱う値:
          - affection_with_doki … ドキドキ補正後の実効好感度
          - doki_level          … その場の高揚段階（0〜4）
          - relationship_level  … 長期的な関係の深さ（0〜100）
          - masking_degree      … ばけばけ度（0〜1）
          - length_mode         … 発話の長さモード（short/normal/long/story/auto）
        """
        emotion_override = emotion_override or {}
        world_state = emotion_override.get("world_state") or {}
        scene_emotion = emotion_override.get("scene_emotion") or {}
        emotion = emotion_override.get("emotion") or {}

        # affection は doki 補正後を優先
        affection = float(
            emotion.get("affection_with_doki", emotion.get("affection", 0.0)) or 0.0
        )
        doki_power = float(emotion.get("doki_power", 0.0) or 0.0)
        doki_level = int(emotion.get("doki_level", 0) or 0)

        # affection_zone があればそれを zone として使う（なければ auto）
        zone = str(emotion.get("affection_zone", "auto") or "auto")

        # relationship / masking（ばけばけ度）
        relationship_level = float(
            emotion.get("relationship_level", emotion.get("relationship", 0.0)) or 0.0
        )
        relationship_stage = str(emotion.get("relationship_stage") or "")
        if not relationship_stage and relationship_level > 0.0:
            relationship_stage = relationship_stage_from_level(relationship_level)

        masking_degree = float(
            emotion.get("masking_degree", emotion.get("masking", 0.0)) or 0.0
        )
        if masking_degree < 0.0:
            masking_degree = 0.0
        if masking_degree > 1.0:
            masking_degree = 1.0

        # world_state から舞台情報
        loc_player = (world_state.get("locations") or {}).get("player")
        location_name = (
            loc_player
            or world_state.get("location_name")
            or world_state.get("player_location")
        )
        time_info = world_state.get("time") or {}
        time_slot = time_info.get("slot") or world_state.get("time_of_day")
        time_str = time_info.get("time_str")

        # 二人きりかどうかの推定
        party_mode = (
            world_state.get("party_mode")
            or (world_state.get("party") or {}).get("mode")
        )
        others_around = world_state.get("others_around")
        is_alone = False
        if party_mode == "alone":
            is_alone = True
        if others_around is False:
            is_alone = True
        if others_around is True:
            is_alone = False

        # masking_defaults による「場所ごとのばけばけ挙動」
        masking_cfg = self._get_masking_defaults()
        unmasked_locs = masking_cfg.get("unmasked_locations", [])
        masked_locs = masking_cfg.get("masked_locations", [])

        loc_key = str(location_name or "").lower()
        is_unmasked_place = bool(
            loc_key and any(tag in loc_key for tag in unmasked_locs)
        )
        is_masked_place = bool(
            loc_key and any(tag in loc_key for tag in masked_locs)
        )

        # 場所に応じた説明用メモ
        masking_env_note = ""
        example_line = ""
        rules = masking_cfg.get("rules") or {}
        raw_example = rules.get("example_line")
        if isinstance(raw_example, str) and raw_example.strip():
            # {PLAYER_NAME} を実際の名前に差し替え
            example_line = raw_example.replace("{PLAYER_NAME}", self.player_name)

        # 「二人きり＋ばけばけ無効」かどうか
        if is_unmasked_place:
            # 自宅／リセ家／部室など → 常に素が出やすい場所
            masking_env_note = (
                "※ 現在は親しい相手とくつろげる場所にいるため、"
                "表情コントロール（ばけばけ度）があってもほとんど働かず、"
                "素直なデレや甘えがそのまま表に出て構いません。"
            )
            if example_line:
                masking_env_note += f"\n  例: 「{example_line}」"
        elif is_masked_place:
            # 学校など人前になりやすい場所
            if is_alone:
                masking_env_note = (
                    "※ 形式上は人目のある場所ですが、いまは実質二人きりなので、"
                    "ばけばけ度はあまり気にせず素直な恋愛感情を見せて構いません。"
                )
                if example_line:
                    masking_env_note += f"\n  例: 「{example_line}」"
            else:
                masking_env_note = (
                    "※ ここは人目のある場所のため、"
                    "ばけばけ度を意識して外見上は一段階落ち着いたトーンで振る舞ってください。"
                    "内心のドキドキや恋愛感情は、仕草や視線、ささやかな言葉ににじませる程度に留めてください。"
                )
        # world_state が無い／マッチしない場合は env_note なし

        # 舞台情報（場所・時間帯）
        location_lines: List[str] = []
        if location_name:
            location_lines.append(f"- 現在の舞台は「{location_name}」。")
        if time_slot or time_str:
            ts = (
                f"{time_slot} / {time_str}"
                if time_slot and time_str
                else (time_slot or time_str)
            )
            location_lines.append(f"- 時間帯は「{ts}」。")

        # 好意ラベル（あれば）
        affection_label = self.get_affection_label(affection)

        # ガイドライン本体（JSON 優先 / 未設定なら簡易デフォルト）
        try:
            guideline = self.build_emotion_control_guideline(
                affection_with_doki=affection,
                doki_level=doki_level,
                mode_current=mode_current,
            )
        except Exception:
            guideline = ""

        if not guideline:
            guideline = self._build_default_guideline(
                affection_with_doki=affection,
                doki_level=doki_level,
                mode_current=mode_current,
            )

        # ばけばけ度数値に基づくデフォルト注意書き
        masking_note = ""
        if masking_degree >= 0.7:
            masking_note = (
                "※ 現在、表情コントロール（ばけばけ度）が高いため、"
                "内心の恋愛感情や高揚をあえて抑え、"
                "外見上は一段階落ち着いたトーンで振る舞ってください。"
            )
        elif masking_degree >= 0.3:
            masking_note = (
                "※ 表情コントロール（ばけばけ度）が中程度のため、"
                "強すぎるデレは少し抑えつつ、"
                "さりげない甘さがにじむ程度に留めてください。"
            )

        # ただし「自宅・リセ家・部室」や「学校でも二人きり」の場合は、
        # 数値的なばけばけ度より環境優先で、masking_note を上書きする。
        if masking_env_note:
            masking_note = masking_env_note

        # 文章量ガイドライン
        length_guideline = self._build_length_guideline(length_mode)

        # ヘッダ組み立て
        header_lines: List[str] = []
        header_lines.append("[感情・関係性プロファイル]")
        header_lines.append(
            f"- 実効好感度 (affection_with_doki): {affection:.2f} "
            f"(zone={zone}, doki_level={doki_level}, doki_power={doki_power:.1f})"
        )
        if affection_label:
            header_lines.append(f"- 好意の解釈: {affection_label}")

        if relationship_level > 0.0:
            header_lines.append(
                f"- 関係レベル (relationship_level): {relationship_level:.1f} / 100"
            )
            if relationship_stage:
                header_lines.append(f"- 関係ステージ: {relationship_stage}")

        if masking_degree > 0.0:
            header_lines.append(
                f"- 表情コントロール（ばけばけ度）: {masking_degree:.2f} "
                "(0=素直 / 1=完全に平静を装う)"
            )

        # 長さモードも一行だけ明示しておく
        header_lines.append(
            f"- 発話の長さモード: {self._normalize_length_mode(length_mode)} "
            "(short/normal/long/story/auto)"
        )

        if location_lines:
            header_lines.extend(location_lines)

        # ドキドキと relationship の違い
        header_lines.append(
            "- 備考: ドキドキ💓はその場の高揚感、relationship_level は長期的な信頼・絆の指標です。"
        )

        if masking_note:
            header_lines.append(masking_note)

        # ブロック連結
        blocks: List[str] = []
        blocks.append("\n".join(header_lines))

        guideline = (guideline or "").strip()
        if guideline:
            blocks.append(guideline)

        length_guideline = (length_guideline or "").strip()
        if length_guideline:
            blocks.append(length_guideline)

        header_block = "\n\n".join(blocks) + "\n"

        if base_system_prompt:
            new_system_prompt = base_system_prompt.rstrip() + "\n\n" + header_block + "\n"
        else:
            new_system_prompt = header_block + "\n"

        return new_system_prompt

    def replace_system_prompt(
        self,
        messages: List[Dict[str, str]],
        new_system_prompt: str,
    ) -> List[Dict[str, str]]:
        """
        messages 内の最初の system を new_system_prompt で置き換える。
        system が無ければ、先頭に追加。
        """
        new_messages = list(messages)
        system_index = None

        for idx, m in enumerate(new_messages):
            if m.get("role") == "system":
                system_index = idx
                break

        system_message = {
            "role": "system",
            "content": new_system_prompt,
        }

        if system_index is not None:
            new_messages[system_index] = system_message
        else:
            new_messages.insert(0, system_message)

        return new_messages

    # ---- EmotionResult → 「感情ヘッダ」（旧 API 互換） ----

    def build_emotion_header(
        self,
        emotion: EmotionResult | None,
        world_state: Dict[str, Any] | None = None,
        scene_emotion: Dict[str, Any] | None = None,
    ) -> str:
        """
        EmotionResult + world_state から
        LLM 用の「感情・関係性ヘッダテキスト」を構築する。
        （古い API 互換用。新規は build_emotion_based_system_prompt を推奨）
        """
        if emotion is None:
            return ""

        world_state = world_state or {}
        scene_emotion = scene_emotion or {}

        # 1) サブクラス完全オーバーライドがあれば優先
        if hasattr(self, "build_emotion_header_hint"):
            try:
                custom = self.build_emotion_header_hint(
                    emotion=emotion,
                    world_state=world_state,
                    scene_emotion=scene_emotion,
                )
                if isinstance(custom, str) and custom.strip():
                    return custom.strip()
            except Exception:
                pass

        # 2) Persona プロファイルから係数取得
        aff_gain = 1.0
        doki_bias = 0.0
        try:
            prof = self.get_emotion_profile() or {}
            aff_gain = float(prof.get("affection_gain", 1.0) or 1.0)
            doki_bias = float(prof.get("doki_bias", 0.0) or 0.0)
        except Exception:
            pass

        # 3) affection_with_doki * gain を 0〜1 にクランプ
        base_aff = float(getattr(emotion, "affection", 0.0) or 0.0)
        aff_with_doki_raw = float(
            getattr(emotion, "affection_with_doki", base_aff) or base_aff
        )
        aff = max(0.0, min(1.0, aff_with_doki_raw * aff_gain))

        # 4) doki_level 0〜4 + bias → [0,4] クランプ
        try:
            dl_raw = int(getattr(emotion, "doki_level", 0) or 0)
        except Exception:
            dl_raw = 0
        dl = int(round(dl_raw + doki_bias))
        if dl < 0:
            dl = 0
        if dl > 4:
            dl = 4

        # 5) affection のゾーン（low/mid/high/extreme）
        aff_zone = affection_to_level(aff)

        # 6) 好意ラベル（あれば）
        affection_label = self.get_affection_label(aff)

        # 7) world_state → 環境ヒント
        location = (
            world_state.get("location_name")
            or world_state.get("player_location")
            or (world_state.get("locations") or {}).get("player")
        )
        time_slot = (
            world_state.get("time_slot")
            or world_state.get("time_of_day")
            or (world_state.get("time") or {}).get("slot")
        )

        scene_hint_parts: List[str] = []
        if location:
            scene_hint_parts.append(f"いま二人は『{location}』付近にいます。")
        if time_slot:
            scene_hint_parts.append(f"時間帯は『{time_slot}』頃です。")
        scene_hint = " ".join(scene_hint_parts).strip()

        # 8) ガイドライン（JSON 優先 / なければ簡易デフォルト）
        try:
            guideline_text = self.build_emotion_control_guideline(
                affection_with_doki=aff,
                doki_level=dl,
                mode_current=getattr(emotion, "mode", "normal"),
            )
        except Exception:
            guideline_text = ""

        if not guideline_text:
            guideline_lines = [
                "[口調・距離感のガイドライン]",
                "1) 特別な感情プロファイルが未設定のため、通常時と同様のトーンで話してください。",
                "2) 相手への基本的な信頼や好意は感じられるように、やわらかな言葉選びを心がけてください。",
            ]
            guideline_text = "\n".join(guideline_lines)

        guideline_text = guideline_text.strip("\n")

        # 9) ヘッダ構築
        header_lines: List[str] = []
        header_lines.append("【感情・関係性プロファイル】")
        header_lines.append(
            f"- 実効好感度（affection_with_doki）: {aff:.2f} "
            f"(zone={aff_zone}, doki_level={dl})"
        )
        if affection_label:
            header_lines.append(f"- 好意の解釈: {affection_label}")
        if scene_hint:
            header_lines.append(f"- 環境: {scene_hint}")

        header_lines.append("")
        header_block = "\n".join(header_lines)

        return header_block + "\n\n" + guideline_text + "\n"

    # --------------------------------------------------
    # デフォルトのガイドライン（JSON 無し時のフォールバック）
    # --------------------------------------------------
    def _build_default_guideline(
        self,
        *,
        affection_with_doki: float,
        doki_level: int,
        mode_current: str,
    ) -> str:
        guideline_lines: List[str] = []
        guideline_lines.append("[口調・距離感のガイドライン]")

        if doki_level >= 4:
            guideline_lines.extend(
                [
                    "1) 結婚を前提にした深い信頼と愛情を前提として、将来への期待がにじむトーンで話してください。",
                    "2) さりげないスキンシップや将来の生活を匂わせる表現を、セリフの中に1つ以上含めてください。",
                    "3) 『ずっとそばにいたい』『本気で大事にしたい』と伝わるニュアンスを自然な描写で入れてください。",
                ]
            )
        elif doki_level == 3:
            guideline_lines.extend(
                [
                    "1) 強い好意と信頼が伝わる、親密で少し独占欲のにじむトーンで話してください。",
                    "2) 距離が近いことや触れそうな距離感を意識した描写を、会話の中にさりげなく混ぜてください。",
                    "3) 相手の体調や気持ちを気遣う言葉を交えつつ、『あなたが大切』というニュアンスを含めてください。",
                ]
            )
        elif doki_level == 2:
            guideline_lines.extend(
                [
                    "1) 付き合い始めのような甘さと緊張感のバランスを意識しながら話してください。",
                    "2) 視線・手の位置・距離感など、少しドキドキしそうな要素を描写に含めてください。",
                    "3) からかい半分・本気半分のような、照れ混じりのセリフを入れても構いません。",
                ]
            )
        elif doki_level == 1:
            guideline_lines.extend(
                [
                    "1) 基本は丁寧で礼儀正しいが、ときどき素直な感情がこぼれるトーンで話してください。",
                    "2) 相手を意識して少しだけ言葉に詰まったり、照れがにじむ描写を入れてください。",
                ]
            )
        else:
            guideline_lines.extend(
                [
                    "1) まだ大きな恋愛感情としては動いていないが、好感や信頼は感じられるフラットなトーンで話してください。",
                    "2) 落ち着いた会話の中に、相手を気遣う一言をさりげなく入れてください。",
                ]
            )

        guideline_lines.append(
            "9) いずれの場合も、キャラクターとして一貫性のある口調と感情表現で返答してください。"
            " 不自然に過剰なベタベタさではなく、その場の状況に合った自然な甘さと距離感を大切にしてください。"
        )

        return "\n".join(guideline_lines)
