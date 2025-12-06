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
    # 公開 API（system prompt / messages）
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
          - （必要なら）追加システム指示や好感度ヒント
          - プレイヤー発話（user_text）
        だけを詰める。
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
    # 感情プロファイル関連（JSON ベース）
    # --------------------------------------------------
    def _get_emotion_profiles(self) -> Dict[str, Any]:
        """
        elf_riseria_da_silva_ja.json 内の emotion_profiles セクションを返す。
        ない場合は空 dict。
        """
        return self.raw.get("emotion_profiles", {}) or {}

    def get_affection_label(self, affection_with_doki: float) -> str:
        """
        affection_with_doki に対応する「好意の解釈」ラベルを JSON から取得する。

        - emotion_profiles.affection_labels: { "0.9": "...", "0.7": "...", ... }
        - affection_with_doki 以上の閾値のうち最大のものを採用（降順で探索）
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
            # うまくパースできなければ諦める
            return ""

        for th in thresholds:
            if affection_with_doki >= th:
                key = f"{th:.1f}".rstrip("0").rstrip(".")  # "0.9" など
                if key in labels:
                    return labels[key]
                # フォーマットずれに備えて元の str も見る
                raw_key = str(th)
                if raw_key in labels:
                    return labels[raw_key]

        # どの閾値も満たさない場合は、最小のものをフォールバックで使う
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
        emotion_prompt_builder から呼ばれる、
        「ドキドキレベルに応じた口調・距離感ガイドライン」の本体。

        中身は JSON (emotion_profiles) によって完全に制御される。
        """
        profiles = self._get_emotion_profiles()
        affection_labels = profiles.get("affection_labels", {}) or {}
        doki_levels = profiles.get("doki_levels", {}) or {}
        mode_overrides = profiles.get("mode_overrides", {}) or {}

        # 好意ラベル
        aff_label = ""
        if affection_labels:
            aff_label = self.get_affection_label(affection_with_doki)

        # doki_level ごとの行リスト
        doki_key = str(int(doki_level))
        doki_lines: List[str] = doki_levels.get(doki_key, []) or []

        # mode 別の追加ガイドライン
        mode_lines: List[str] = mode_overrides.get(str(mode_current), []) or []

        lines: List[str] = []
        lines.append("[リセリア用・口調と距離感ガイドライン]")

        if aff_label:
            lines.append(f"- 現在の好意の解釈: {aff_label}")

        if doki_lines:
            lines.extend(doki_lines)

        if mode_lines:
            lines.append("")
            lines.append("[モード別ガイドライン]")
            lines.extend(mode_lines)

        if not doki_lines and not mode_lines and not aff_label:
            # JSON が未設定の場合の最低限フォールバック
            lines.append(
                "※ 感情プロファイルが未設定のため、通常時とほぼ同じトーンで話してください。"
            )

        return "\n".join(lines)
