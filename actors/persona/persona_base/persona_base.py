# actors/persona/persona_base/persona_base.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
import json


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

        # system_prompt 系（base + public/private suffix を JSON から構成）
        (
            base_sp,
            public_suffix,
            private_suffix,
        ) = self._build_system_prompts_from_json(player_name=player_name)

        self.system_prompt_base: str = base_sp
        self.system_prompt_public_suffix: str = public_suffix
        self.system_prompt_private_suffix: str = private_suffix

        # 互換用：古いコードが self.system_prompt を読んでも動くようにベースを入れておく
        self.system_prompt: str = self.system_prompt_base

    # --------------------------------------------------
    # Streamlit(llm_meta) への保険書き込み
    # --------------------------------------------------
    def _try_set_llm_meta(self, key: str, value: Any) -> None:
        """
        Persona 側から st.session_state["llm_meta"] を安全に更新するためのヘルパ。
        Streamlit 非依存で動かしたい場面もあるため、import は try する。
        """
        try:
            import streamlit as st  # type: ignore
        except Exception:
            return

        try:
            meta = st.session_state.get("llm_meta")
            if not isinstance(meta, dict):
                meta = {}
                st.session_state["llm_meta"] = meta
            meta[key] = value
        except Exception:
            # ここで例外を出すと会話全体が崩れるので絶対落とさない
            return

    # --------------------------------------------------
    # JSON ロード
    # --------------------------------------------------
    def _load_json(self) -> Dict[str, Any]:
        """
        /actors/persona/persona_datas/{JSON_NAME} を読む。
        """
        if not self.JSON_NAME:
            return {}

        here = Path(__file__).resolve().parent  # actors/persona/persona_base/
        root = here.parent  # actors/persona/
        json_path = root / "persona_datas" / self.JSON_NAME

        if not json_path.exists():
            print(f"[PersonaBase] JSON not found: {json_path}")
            return {}

        text = json_path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except Exception as e:
            print(f"[PersonaBase] JSON parse error at {json_path}: {e}")
            return {}

        if isinstance(data, dict):
            return data
        return {}

    # --------------------------------------------------
    # system_prompt 自動構成
    # --------------------------------------------------
    @staticmethod
    def _join_lines(lines: List[Any]) -> str:
        parts: List[str] = []
        for x in lines:
            s = str(x).strip()
            if s:
                parts.append(s)
        return "\n".join(parts)

    def _build_system_prompts_from_json(
        self,
        *,
        player_name: str,
    ) -> tuple[str, str, str]:
        """
        JSON から system_prompt_base / public_suffix / private_suffix を決定する。

        優先順位:
        1) conversation_rules.roleplay_instructions_common/public/private が定義されていれば、
           それを使って base ＋ suffix を構築する。
        2) そうでなければ、従来どおり system_prompt と
           system_prompt_public_suffix / system_prompt_private_suffix をそのまま使う。
        """

        raw_sp = str(self.raw.get("system_prompt", "") or "")

        conv = self.raw.get("conversation_rules") or {}
        if not isinstance(conv, dict):
            conv = {}

        common_list = conv.get("roleplay_instructions_common") or []
        public_list = conv.get("roleplay_instructions_public") or []
        private_list = conv.get("roleplay_instructions_private") or []
        flat_list = conv.get("roleplay_instructions") or []

        has_segmented = bool(common_list or public_list or private_list)

        if has_segmented:
            # ========== 新スタイル：common / public / private を使う ==========
            common_text = self._join_lines(common_list)
            public_suffix = self._join_lines(public_list)
            private_suffix = self._join_lines(private_list)
            flat_text = self._join_lines(flat_list)

            # ベースは「元の system_prompt（あれば）」＋ common
            if raw_sp and common_text:
                base = raw_sp.strip() + "\n\n" + common_text
            elif raw_sp:
                base = raw_sp.strip()
            elif common_text:
                base = common_text
            elif flat_text:
                base = flat_text
            else:
                base = ""
        else:
            # ========== 旧スタイル互換：素の system_prompt + suffix フィールド ==========
            base = raw_sp
            public_suffix = str(self.raw.get("system_prompt_public_suffix", "") or "")
            private_suffix = str(self.raw.get("system_prompt_private_suffix", "") or "")

        # プレイヤー名差し替え
        for token in ("{PLAYER_NAME}", "{player_name}"):
            base = base.replace(token, player_name)
            public_suffix = public_suffix.replace(token, player_name)
            private_suffix = private_suffix.replace(token, player_name)

        return base, public_suffix, private_suffix

    # --------------------------------------------------
    # system prompt / messages
    # --------------------------------------------------
    def get_system_prompt(self) -> str:
        """Actor / AnswerTalker などから参照される想定のヘルパ。"""
        return self.system_prompt_base

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
        system_parts: List[str] = [self.system_prompt_base]

        if extra_system_hint:
            extra = extra_system_hint.strip()
            if extra:
                system_parts.append(extra)

        if affection_hint:
            ah = affection_hint.strip()
            if ah:
                system_parts.append(ah)

        system_text = "\n\n".join(system_parts)

        # ★最小修正：ここで「少なくともこのsystemは使った」を必ず残す
        # AnswerTalker側が保存し忘れても View で表示できる
        self._try_set_llm_meta("system_prompt_used", system_text)

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
        from actors.persona.persona_base.build_default_guideline import (
            build_default_guideline,
        )

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

        # JSON が空の場合はデフォルトガイドラインにフォールバック
        if not doki_lines and not mode_lines and not aff_label:
            return build_default_guideline(
                affection_with_doki=affection_with_doki,
                doki_level=doki_level,
                mode_current=mode_current,
            )

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

        実装本体は actors/persona/persona_base/build_emotion_based_system_prompt_core.py に分離。
        """
        from actors.persona.persona_base.build_emotion_based_system_prompt_core import (
            build_emotion_based_system_prompt_core,
        )

        sp = build_emotion_based_system_prompt_core(
            persona=self,
            base_system_prompt=base_system_prompt,
            emotion_override=emotion_override,
            mode_current=mode_current,
            length_mode=length_mode,
        )

        # ★最小修正：最終的に使う system_prompt を必ず残す（View に出すため）
        self._try_set_llm_meta("system_prompt_used", sp)

        return sp

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

    # ---- EmotionResult → 「感情ヘッダ」 ----
    def build_emotion_header(
        self,
        emotion: Any | None,
        world_state: Dict[str, Any] | None = None,
        scene_emotion: Dict[str, Any] | None = None,
    ) -> str:
        """
        EmotionResult + world_state から
        LLM 用の「感情・関係性ヘッダテキスト」を構築する。

        実装本体は actors/persona/persona_base/build_emotion_header.py に分離。
        """
        from actors.persona.persona_base.build_emotion_header import (
            build_emotion_header_core,
        )

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
        実装本体は actors/persona/persona_base/build_default_guideline.py。
        """
        from actors.persona.persona_base.build_default_guideline import (
            build_default_guideline,
        )

        return build_default_guideline(
            affection_with_doki=affection_with_doki,
            doki_level=doki_level,
            mode_current=mode_current,
        )
