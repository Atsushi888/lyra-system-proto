# actors/scene_ai.py
from __future__ import annotations

from typing import Any, Dict, MutableMapping, Optional


class SceneAI:
    """
    SceneChanger / UI 側が st.session_state に書き込んだ
    「現在シーン情報」を読み取り、ロジック層に渡すための薄いラッパ。

    想定する state のキー（必要に応じて scene_changer.py 側と揃えてください）:

      - scene_current: str
          現在のシーンID
            例: "exit", "town", "road", "ice_cave"

      - scene_label: str
          プレイヤー向けの表示名
            例: "会話を終了する", "街", "街道筋", "封印の氷窟"

      - scene_emotion_bonus: Dict[str, float]
          感情補正値。
          keys: affection / arousal / tension / anger / sadness / excitement

    上記が無い場合は、デフォルト値（ID=town / ラベル=街 / ボーナス0）で補完する。
    """

    def __init__(self, state: Optional[MutableMapping[str, Any]] = None) -> None:
        # state には st.session_state を渡しておく想定。
        # 渡されなかった場合は None → すべてデフォルト扱い。
        self.state = state

    # --------------------------------------------------------------
    # 公開 API
    # --------------------------------------------------------------
    def get_current_scene_info(self) -> Dict[str, Any]:
        """
        現在シーンの情報を辞書で返す。

        戻り値のフォーマットは MixerAI が期待している形に揃える:

        {
          "scene_id": "town",
          "label": "街",
          "emotion_bonus": {
              "affection": 0.0,
              "arousal": 0.0,
              "tension": 0.0,
              "anger": 0.0,
              "sadness": 0.0,
              "excitement": 0.0,
          },
        }
        """
        scene_id = "town"
        label = self._default_label(scene_id)
        bonus: Dict[str, float] = {}

        st = self.state
        if st is not None:
            # scene_current
            try:
                cur = st.get("scene_current")
            except Exception:
                cur = None
            if isinstance(cur, str) and cur.strip():
                scene_id = cur.strip()

            # scene_label（無ければ ID から推定）
            try:
                lbl = st.get("scene_label")
            except Exception:
                lbl = None
            if isinstance(lbl, str) and lbl.strip():
                label = lbl.strip()
            else:
                label = self._default_label(scene_id)

            # emotion_bonus
            try:
                raw_bonus = st.get("scene_emotion_bonus")
            except Exception:
                raw_bonus = None

            if isinstance(raw_bonus, dict):
                bonus = self._normalize_bonus(raw_bonus)

        # 最終的な dict を組み立て
        info = {
            "scene_id": scene_id,
            "label": label,
            "emotion_bonus": {
                "affection": bonus.get("affection", 0.0),
                "arousal": bonus.get("arousal", 0.0),
                "tension": bonus.get("tension", 0.0),
                "anger": bonus.get("anger", 0.0),
                "sadness": bonus.get("sadness", 0.0),
                "excitement": bonus.get("excitement", 0.0),
            },
        }
        return info

    # --------------------------------------------------------------
    # 内部ヘルパ
    # --------------------------------------------------------------
    @staticmethod
    def _default_label(scene_id: str) -> str:
        """
        シーンID → ラベル のデフォルト対応表。

        scene_changer.py 側で好きに上書きしてもらってOK。
        """
        mapping = {
            "exit": "会話を終了する",
            "town": "街",
            "road": "街道筋",
            "ice_cave": "封印の氷窟",
        }
        return mapping.get(scene_id, scene_id)

    @staticmethod
    def _normalize_bonus(raw: Dict[str, Any]) -> Dict[str, float]:
        """
        scene_emotion_bonus に入っている値を float に正規化。
        """
        keys = [
            "affection",
            "arousal",
            "tension",
            "anger",
            "sadness",
            "excitement",
        ]
        out: Dict[str, float] = {}
        for k in keys:
            v = raw.get(k, 0.0)
            try:
                out[k] = float(v)
            except Exception:
                out[k] = 0.0
        return out
