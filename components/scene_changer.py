# components/scene_changer.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any

import os
import json
import streamlit as st


# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------

DEFAULT_SCENE_BONUS_DIR = "actors/scene_bonus"


def get_scene_bonus_dir() -> str:
    """
    シーンボーナス JSON を保存するディレクトリを返す。

    - st.secrets["SCENE_BONUS_DIR"] があれば優先
    - なければ DEFAULT_SCENE_BONUS_DIR
    """
    base = DEFAULT_SCENE_BONUS_DIR
    try:
        if "SCENE_BONUS_DIR" in st.secrets:
            base = str(st.secrets["SCENE_BONUS_DIR"])
    except Exception:
        pass

    os.makedirs(base, exist_ok=True)
    return base


# ---------------------------------------------------------
# モデル定義
# ---------------------------------------------------------

@dataclass
class SceneBonusConfig:
    scene_id: str
    label: str
    emotion_bonus: Dict[str, float]

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "SceneBonusConfig":
        scene_id = str(data.get("scene_id", "unknown"))
        label = str(data.get("label", scene_id))
        eb = data.get("emotion_bonus") or {}
        # 必要なキーだけ抽出し、float 化
        bonus: Dict[str, float] = {}
        for k in ["affection", "arousal", "tension", "anger", "sadness", "excitement"]:
            v = eb.get(k, 0.0)
            try:
                bonus[k] = float(v)
            except Exception:
                bonus[k] = 0.0
        return cls(scene_id=scene_id, label=label, emotion_bonus=bonus)


# ---------------------------------------------------------
# SceneChanger 本体
# ---------------------------------------------------------

class SceneChanger:
    """
    シーン移動＋シーンごとの感情ボーナス編集 UI。

    - シーン選択（会話終了 / 街 / 街道筋 / 封印の氷窟）
    - 感情ボーナスのスライダー編集
    - JSON への保存 / JSON からの再読込
    - 決定した内容を st.session_state に反映：
        - scene_current: str
        - scene_label: str
        - scene_emotion_bonus: Dict[str, float]
    """

    # 固定シーン定義
    SCENES: Dict[str, str] = {
        "end": "会話を終了する",
        "town": "街",
        "road": "街道筋",
        "ice_cave": "封印の氷窟",
    }

    def __init__(self, *, session_prefix: str = "scene_") -> None:
        self.session_prefix = session_prefix
        self.bonus_dir = get_scene_bonus_dir()

    # -----------------------------
    # JSON セーブ／ロード
    # -----------------------------

    def _scene_json_path(self, scene_id: str) -> str:
        return os.path.join(self.bonus_dir, f"{scene_id}.json")

    def load_bonus_for_scene(self, scene_id: str) -> SceneBonusConfig:
        """
        指定シーンのボーナス設定を JSON から読み込む。
        無ければデフォルト 0.0 で作成。
        """
        label = self.SCENES.get(scene_id, scene_id)
        path = self._scene_json_path(scene_id)

        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cfg = SceneBonusConfig.from_json_dict(data)
                # label はコード側優先（日本語名を変えたときのため）
                cfg.label = label
                return cfg
            except Exception:
                pass

        # デフォルト（全部 0）
        return SceneBonusConfig(
            scene_id=scene_id,
            label=label,
            emotion_bonus={
                "affection": 0.0,
                "arousal": 0.0,
                "tension": 0.0,
                "anger": 0.0,
                "sadness": 0.0,
                "excitement": 0.0,
            },
        )

    def save_bonus_for_scene(self, cfg: SceneBonusConfig) -> None:
        """
        指定シーンのボーナス設定を JSON に保存。
        """
        path = self._scene_json_path(cfg.scene_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg.to_json_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            st.error(f"シーン設定の保存に失敗しました: {e}")

    # -----------------------------
    # session_state ヘルパ
    # -----------------------------

    def _set_state(self, key: str, value: Any) -> None:
        st.session_state[f"{self.session_prefix}{key}"] = value

    def _get_state(self, key: str, default: Any = None) -> Any:
        return st.session_state.get(f"{self.session_prefix}{key}", default)

    # -----------------------------
    # メイン UI
    # -----------------------------

    def render(self) -> None:
        st.markdown("## 🚶‍♀️ シーン移動 / シーン感情ボーナス設定")
        st.caption(
            "ここで選択したシーンと感情ボーナスは、MixerAI / AnswerTalker 経由で\n"
            "フローリアの感情状態に影響を与えます（将来拡張を含む）。"
        )

        # 現在のシーン ID（無ければ "town"）
        current_scene_id = self._get_state("current", "town")

        # シーン選択
        scene_ids = list(self.SCENES.keys())
        scene_labels = [self.SCENES[sid] for sid in scene_ids]

        idx_default = max(scene_ids.index(current_scene_id), 0) if current_scene_id in scene_ids else 0

        selected_label = st.selectbox(
            "移動先シーンを選んでください：",
            options=scene_labels,
            index=idx_default,
        )
        # ラベル→ID 逆引き
        scene_id = scene_ids[scene_labels.index(selected_label)]
        scene_label = self.SCENES[scene_id]

        # JSON からボーナスをロード（UI 初期値用）
        cfg = self.load_bonus_for_scene(scene_id)
        bonus = cfg.emotion_bonus

        st.markdown(f"### シーン: **{scene_label}**  (`{scene_id}`)")
        st.caption("このシーンにいる間、フローリアの感情値に加算されるボーナス／ペナルティ。")

        cols1 = st.columns(3)
        cols2 = st.columns(3)

        with cols1[0]:
            affection = st.slider(
                "affection（好意）",
                -1.0, 1.0, float(bonus.get("affection", 0.0)),
                step=0.05,
            )
        with cols1[1]:
            arousal = st.slider(
                "arousal（性的興奮）",
                -1.0, 1.0, float(bonus.get("arousal", 0.0)),
                step=0.05,
            )
        with cols1[2]:
            tension = st.slider(
                "tension（緊張）",
                -1.0, 1.0, float(bonus.get("tension", 0.0)),
                step=0.05,
            )

        with cols2[0]:
            anger = st.slider(
                "anger（怒り）",
                -1.0, 1.0, float(bonus.get("anger", 0.0)),
                step=0.05,
            )
        with cols2[1]:
            sadness = st.slider(
                "sadness（悲しみ）",
                -1.0, 1.0, float(bonus.get("sadness", 0.0)),
                step=0.05,
            )
        with cols2[2]:
            excitement = st.slider(
                "excitement（高揚）",
                -1.0, 1.0, float(bonus.get("excitement", 0.0)),
                step=0.05,
            )

        # 最新値で cfg を更新
        cfg.emotion_bonus = {
            "affection": float(affection),
            "arousal": float(arousal),
            "tension": float(tension),
            "anger": float(anger),
            "sadness": float(sadness),
            "excitement": float(excitement),
        }
        cfg.label = scene_label

        st.markdown("---")

        # JSON セーブ / ロードボタン
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        save_clicked = False
        reload_clicked = False

        with col_btn1:
            if st.button("💾 このシーン設定を JSON に保存", use_container_width=True):
                self.save_bonus_for_scene(cfg)
                save_clicked = True
        with col_btn2:
            if st.button("🔁 JSON からこのシーン設定を再読込", use_container_width=True):
                cfg = self.load_bonus_for_scene(scene_id)
                reload_clicked = True
        with col_btn3:
            if st.button("✅ この内容でシーンを確定", use_container_width=True):
                # st.session_state に反映
                self._set_state("current", scene_id)
                self._set_state("label", scene_label)
                self._set_state("emotion_bonus", cfg.emotion_bonus)
                st.success(f"シーンを『{scene_label}』に確定しました。")

        if save_clicked:
            st.success(f"シーン『{scene_label}』のボーナス設定を保存しました。")

        if reload_clicked:
            st.info(
                f"シーン『{scene_label}』のボーナス設定を JSON から再読込しました。\n"
                "※ スライダーの値は次回の再描画時に反映されます。"
            )

        # 最後に、常に「現在の状態」を session_state に同期
        self._set_state("current", scene_id)
        self._set_state("label", scene_label)
        self._set_state("emotion_bonus", cfg.emotion_bonus)
