# actors/scene/scene_manager.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any
from datetime import datetime
import json
import os

import streamlit as st


@dataclass
class SceneManager:
    """
    場所 × 時間帯の感情補正テーブルを管理するクラス。
    Streamlit UI も内蔵しており、単体でエディタとして機能する。
    """

    # JSON 保存先（あなたの指定：actors/scene/scene_bonus/）
    path: str = "actors/scene/scene_bonus/scene_emotion_map.json"

    locations: Dict[str, Dict[str, float]] = field(default_factory=dict)
    times: Dict[str, Dict[str, float]] = field(default_factory=dict)
    combined: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ====== 基本 I/O ======
    def load(self) -> None:
        """JSON ファイルから Scene 情報を読み込む。存在しない場合は初期化。"""
        if not os.path.exists(self.path):
            self._init_default()
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            self._init_default()
            return

        self.locations = {
            name: v.get("base", {})
            for name, v in data.get("locations", {}).items()
        }
        self.times = {
            name: v.get("base", {})
            for name, v in data.get("times", {}).items()
        }
        self.combined = data.get("combined", {})

        if not self.locations and not self.times:
            self._init_default()

    def save(self) -> None:
        """現在の Scene 情報を JSON に保存。"""
        data = {
            "meta": {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
            "locations": {
                name: {"base": vals}
                for name, vals in self.locations.items()
            },
            "times": {
                name: {"base": vals}
                for name, vals in self.times.items()
            },
            "combined": self.combined,
        }

        dir_name = os.path.dirname(self.path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _init_default(self) -> None:
        """初期用テンプレートデータ."""
        self.locations = {
            "通学路":          {"affection": 0.1,  "arousal": -0.1, "tension": -0.1},
            "学食":            {"affection": 0.2,  "arousal": -0.2, "tension": -0.1},
            "駅前":            {"affection": 0.15, "arousal": 0.0,  "tension": 0.0},
            "プレイヤーの部屋": {"affection": 0.25, "arousal": 0.1,  "tension": -0.1},
            "プール":          {"affection": 0.3,  "arousal": 0.2,  "tension": 0.1},
        }

        self.times = {
            "morning":      {"affection": 0.1,  "arousal": -0.2, "tension": 0.0},
            "lunch":        {"affection": 0.15, "arousal": -0.1, "tension": -0.05},
            "after_school": {"affection": 0.25, "arousal": 0.2,  "tension": 0.1},
            "night":        {"affection": 0.2,  "arousal": 0.15, "tension": 0.05},
        }

        self.rebuild_combined()

    # ====== 合成ロジック ======
    def _combine_vector(
        self,
        loc_vec: Dict[str, float],
        time_vec: Dict[str, float],
        w_loc: float = 0.6,
        w_time: float = 0.4,
    ) -> Dict[str, float]:
        """場所と時間帯の感情ベクトルを重み付きで合成する。"""
        keys = set(loc_vec.keys()) | set(time_vec.keys())
        out: Dict[str, float] = {}
        denom = max(w_loc + w_time, 1e-6)

        for k in keys:
            lv = float(loc_vec.get(k, 0.0))
            tv = float(time_vec.get(k, 0.0))
            out[k] = (w_loc * lv + w_time * tv) / denom

        return out

    def rebuild_combined(self) -> None:
        """全組み合わせの感情補正値を再計算."""
        combined: Dict[str, Dict[str, float]] = {}
        for loc_name, loc_vec in self.locations.items():
            for time_name, time_vec in self.times.items():
                key = f"{loc_name}@{time_name}"
                combined[key] = self._combine_vector(loc_vec, time_vec)

        self.combined = combined

    def get_for(self, location: str, time_of_day: str) -> Dict[str, float]:
        """SceneAIやMixerAIから使用するランタイムAPI."""
        key = f"{location}@{time_of_day}"
        if key in self.combined:
            return dict(self.combined[key])

        loc_vec = self.locations.get(location, {})
        time_vec = self.times.get(time_of_day, {})
        return self._combine_vector(loc_vec, time_vec)

    # ====== Streamlit UI (SceneManagerView の本体) ======
    def render(self) -> None:
        """SceneManager エディタ UI."""
        st.markdown("## 🎚 Scene Emotion Manager")
        st.caption(f"保存先: `{self.path}`")

        if not self.locations and not self.times:
            self._init_default()

        # --- 場所別 ---
        st.markdown("### 🏙 ロケーション別ベース補正")

        for loc_name in list(self.locations.keys()):
            st.markdown(f"**場所: {loc_name}**")
            vec = self.locations.setdefault(
                loc_name,
                {"affection": 0.0, "arousal": 0.0, "tension": 0.0},
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                vec["affection"] = st.slider(
                    f"{loc_name} / affection",
                    -1.0, 1.0,
                    float(vec.get("affection", 0.0)), 0.05,
                    key=f"loc_{loc_name}_affection",
                )
            with col2:
                vec["arousal"] = st.slider(
                    f"{loc_name} / arousal",
                    -1.0, 1.0,
                    float(vec.get("arousal", 0.0)), 0.05,
                    key=f"loc_{loc_name}_arousal",
                )
            with col3:
                vec["tension"] = st.slider(
                    f"{loc_name} / tension",
                    -1.0, 1.0,
                    float(vec.get("tension", 0.0)), 0.05,
                    key=f"loc_{loc_name}_tension",
                )

            st.markdown("---")

        # --- 時間帯別 ---
        st.markdown("### 🕒 時間帯別ベース補正")

        for time_name in list(self.times.keys()):
            st.markdown(f"**時間帯: {time_name}**")
            vec = self.times.setdefault(
                time_name,
                {"affection": 0.0, "arousal": 0.0, "tension": 0.0},
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                vec["affection"] = st.slider(
                    f"{time_name} / affection",
                    -1.0, 1.0,
                    float(vec.get("affection", 0.0)), 0.05,
                    key=f"time_{time_name}_affection",
                )
            with col2:
                vec["arousal"] = st.slider(
                    f"{time_name} / arousal",
                    -1.0, 1.0,
                    float(vec.get("arousal", 0.0)), 0.05,
                    key=f"time_{time_name}_arousal",
                )
            with col3:
                vec["tension"] = st.slider(
                    f"{time_name} / tension",
                    -1.0, 1.0,
                    float(vec.get("tension", 0.0)), 0.05,
                    key=f"time_{time_name}_tension",
                )

            st.markdown("---")

        # --- 保存 ---
        if st.button("💾 再計算して保存", type="primary"):
            self.rebuild_combined()
            self.save()
            st.success("Scene 情報を保存しました。")

        # --- デバッグプレビュー ---
        with st.expander("🧪 combined preview"):
            st.json(self.combined)
