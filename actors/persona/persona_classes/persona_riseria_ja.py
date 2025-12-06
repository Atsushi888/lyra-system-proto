# actors/persona/persona_classes/persona_riseria_ja.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class Persona:
    """
    リセリア・ダ・シルヴァ用 Persona。

    - /actors/persona/persona_datas/elf_riseria_da_silva_ja.json を読み込み
    - {PLAYER_NAME} プレースホルダを差し替えた system_prompt を提供
    - Actor / AnswerTalker から呼ばれる build_messages() を実装
    """

    JSON_NAME = "elf_riseria_da_silva_ja.json"

    def __init__(self, player_name: str = "アツシ") -> None:
        self.player_name = player_name

        data = self._load_json()

        # JSON の生データ
        self.raw: Dict[str, Any] = data

        # 基本プロファイル
        self.id: str = data.get("id", "elf_riseria_da_silva_ja")
        self.display_name: str = data.get("display_name", "リセリア・ダ・シルヴァ")
        self.short_name: str = data.get("short_name", "リセ")

        # AnswerTalker / PersonaAI が見る用の char_id
        # （なければ "default" 扱いになってしまうので、id と揃えておく）
        self.char_id: str = self.id

        # system_prompt 内の {PLAYER_NAME} を差し替え
        base_sp = data.get("system_prompt", "")
        self.system_prompt: str = base_sp.replace("{PLAYER_NAME}", player_name)

    # --------------------------------------------------
    # JSON ロード
    # --------------------------------------------------
    def _load_json(self) -> Dict[str, Any]:
        """
        /actors/persona/persona_datas/elf_riseria_da_silva_ja.json
        から Persona 定義を読み込む。
        """
        here = Path(__file__).resolve().parent
        # .../actors/persona/persona_classes/ → persona_datas/ へ
        json_path = here.parent / "persona_datas" / self.JSON_NAME

        text = json_path.read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        return {}

    # --------------------------------------------------
    # 公開 API（既存）
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
        Actor → AnswerTalker に渡すための messages を構築する。

        いまのところシンプルに
          - リセリアの system_prompt
          - （必要なら）好感度ヒントや追加システム指示
          - プレイヤー発話（user_text）
        だけを詰める。

        conversation_log / world_state は将来拡張用の受け皿として受け取るだけで、
        現段階では使わない（互換性確保のために引数だけ受けて捨てる）。
        """
        system_parts: List[str] = [self.system_prompt]

        if extra_system_hint:
            extra_system_hint = extra_system_hint.strip()
            if extra_system_hint:
                system_parts.append(extra_system_hint)

        if affection_hint:
            affection_hint = affection_hint.strip()
            if affection_hint:
                system_parts.append(affection_hint)

        system_text = "\n\n".join(system_parts)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        return messages

    # --------------------------------------------------
    # 公開 API（感情プロファイル関連・新規）
    # --------------------------------------------------
    def get_emotion_profile(self) -> Dict[str, Any]:
        """
        JSON 内の affection_levels セクションをそのまま返す。

        期待される構造:
        {
            "low": {"description": str, "speech_hints": [str, ...]},
            "mid": {...},
            "high": {...},
            "extreme": {...},
        }
        """
        return self.raw.get("affection_levels", {}) or {}

    def _estimate_stage_code(
        self,
        affection_with_doki: float,
        doki_level: int,
    ) -> str:
        """
        affection + doki_level から、low/mid/high/extreme のどれに寄せるかをざっくり決める。
        """
        # ベースは affection のしきい値
        if affection_with_doki >= 0.80:
            base = "high"
        elif affection_with_doki >= 0.55:
            base = "mid"
        elif affection_with_doki >= 0.30:
            base = "low"
        else:
            base = "low"

        # doki_level の影響を上乗せ
        # 0: そのまま
        # 1: low→mid くらいまで
        # 2: mid 以上を積極的に
        # 3: high 以上を強制
        # 4: extreme 固定
        order = ["low", "mid", "high", "extreme"]
        idx = order.index(base)

        if doki_level >= 4:
            return "extreme"
        elif doki_level == 3:
            # 最低でも high まで引き上げ
            return "high" if idx < 2 else order[min(idx + 1, 3)]
        elif doki_level == 2:
            # mid 以上を狙う
            return "mid" if idx < 1 else order[min(idx + 1, 3)]
        elif doki_level == 1:
            # ほんのり上乗せ
            return order[min(idx + 1, 3)]
        else:
            return base

    def build_emotion_control_guideline(
        self,
        *,
        affection_with_doki: float,
        doki_level: int,
        mode_current: str = "normal",
    ) -> str:
        """
        emotion_prompt_builder から呼ばれる、
        「この Persona 専用の口調・距離感ガイドライン」を返す。

        - elf_riseria_da_silva_ja.json の affection_levels.*
          description / speech_hints をベースに組み立てる。
        - doki_level に応じてステージを補正する。
        """
        levels = self.get_emotion_profile()
        stage_code = self._estimate_stage_code(
            affection_with_doki=affection_with_doki,
            doki_level=doki_level,
        )

        stage = levels.get(stage_code, {})
        description: str = stage.get("description", "")
        speech_hints: List[str] = stage.get("speech_hints", []) or []

        lines: List[str] = []
        lines.append("[口調・距離感のガイドライン（リセリア専用）]")

        if description:
            lines.append(f"- 推定ステージ: {stage_code.upper()}  ({description})")
        else:
            lines.append(f"- 推定ステージ: {stage_code.upper()}")

        if speech_hints:
            lines.append("")
            lines.append("【セリフ／振る舞いのヒント】")
            for i, hint in enumerate(speech_hints, start=1):
                lines.append(f"{i}) {hint}")

        # erotic モード時は、R18に踏み込まない範囲での追加指示だけ足しておく
        if str(mode_current) == "erotic":
            lines.append("")
            lines.append(
                "【モード補足（erotic）】"
            )
            lines.append(
                "※ 直接的・露骨な描写には踏み込まず、"
                "甘いロマンスや少し踏み込んだスキンシップの“匂わせ”に留めてください。"
            )

        # どのステージでも共通の締め
        lines.append("")
        lines.append(
            "※ いずれの場合も、リセリアとして一貫したコウハイ口調を維持しつつ、"
            "相手を深く信頼していることがセリフや仕草から自然に伝わるように表現してください。"
        )

        return "\n".join(lines)
