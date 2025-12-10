from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from actors.emotion_ai import EmotionResult
from actors.persona.persona_base.build_default_guideline import build_default_guideline
from actors.persona.persona_base.build_emotion_based_system_prompt_core import (
    build_emotion_based_system_prompt_core,
)
from actors.persona.persona_base.build_emotion_header import build_emotion_header_core


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

        # 旧実装互換用の persona_id エイリアス
        # （古いコードが persona.persona_id を参照しても動くようにする）
        self.persona_id: str = self.id

        self.display_name: str = self.raw.get("display_name", self.id)
        self.short_name: str = self.raw.get("short_name", self.display_name)
    
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

        実装本体は actors/persona/build_emotion_based_system_prompt_core.py に分離。
        """
        return build_emotion_based_system_prompt_core(
            persona=self,
            base_system_prompt=base_system_prompt,
            emotion_override=emotion_override,
            mode_current=mode_current,
            length_mode=length_mode,
        )

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

        実装本体は actors/persona/build_emotion_header.py に分離。
        """
        return build_emotion_header_core(
            persona=self,
            emotion=emotion,
            world_state=world_state,
            scene_emotion=scene_emotion,
        )

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
        """
        互換性維持用ラッパー。
        実装本体は actors/persona/build_default_guideline.py。
        """
        return build_default_guideline(
            affection_with_doki=affection_with_doki,
            doki_level=doki_level,
            mode_current=mode_current,
        )
