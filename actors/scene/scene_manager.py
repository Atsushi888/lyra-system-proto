# actors/scene/scene_manager.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime, time
import json
import os

import streamlit as st


@dataclass
class SceneManager:
    """
    場所ごとに「一日の時間帯スロット」と「感情補正ベクトル」を持つマネージャ。

    JSON構造（v2.0-slot の例）:
    {
      "meta": {
        "version": "2.0-slot",
        "updated_at": "...",
        "dimensions": ["affection", "arousal", "tension"]
      },
      "time_slots": {
        "morning": { "start": "07:00", "end": "09:00" },
        ...
      },
      "locations": {
        "通学路": {
          "slots": {
            "morning": { "emotions": { "affection": 0.1, ... } },
            ...
          }
        },
        ...
      }
    }
    """

    path: str = "actors/scene/scene_bonus/scene_emotion_map.json"

    # 感情次元（UI はこのリストに従ってスライダーを出す）
    dimensions: List[str] = field(
        default_factory=lambda: ["affection", "arousal", "tension"]
    )

    # "morning" → {"start": "07:00", "end": "09:00"}
    time_slots: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # "通学路" → {"slots": { "morning": {"emotions": {...}}, ... }}
    locations: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ====== 基本 I/O ======
    def load(self) -> None:
        """JSON から読み込む。なければデフォルト初期化。"""
        if not os.path.exists(self.path):
            self._init_default()
            return

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            self._init_default()
            return

        meta = data.get("meta", {})
        version = meta.get("version", "")

        # v2 以外は互換を考えず初期化してしまう（初期段階なので割り切り）
        if version != "2.0-slot":
            self._init_default()
            return

        self.dimensions = meta.get("dimensions", self.dimensions)
        self.time_slots = data.get("time_slots", {})
        self.locations = data.get("locations", {})

        # セーフティ：最低限の値がなければ初期化
        if not self.time_slots or not self.locations:
            self._init_default()

    def save(self) -> None:
        """現在の Scene 情報を JSON に保存。"""
        data = {
            "meta": {
                "version": "2.0-slot",
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "dimensions": self.dimensions,
            },
            "time_slots": self.time_slots,
            "locations": self.locations,
        }

        dir_name = os.path.dirname(self.path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _init_default(self) -> None:
        """通学路・学食などを前提にしたデフォルトセット。"""
        self.dimensions = ["affection", "arousal", "tension"]

        self.time_slots = {
            "morning":      {"start": "07:00", "end": "09:00"},
            "lunch":        {"start": "12:00", "end": "13:00"},
            "after_school": {"start": "16:00", "end": "19:00"},
            "night":        {"start": "20:00", "end": "23:30"},
        }

        self.locations = {
            "通学路": {
                "slots": {
                    "morning": {
                        "emotions": {"affection": 0.10, "arousal": -0.10, "tension": -0.10}
                    },
                    "after_school": {
                        "emotions": {"affection": 0.25, "arousal": 0.20, "tension": 0.10}
                    },
                }
            },
            "学食": {
                "slots": {
                    "lunch": {
                        "emotions": {"affection": 0.20, "arousal": -0.20, "tension": -0.10}
                    }
                }
            },
            "駅前": {
                "slots": {
                    "after_school": {
                        "emotions": {"affection": 0.15, "arousal": 0.00, "tension": 0.00}
                    },
                    "night": {
                        "emotions": {"affection": 0.18, "arousal": 0.10, "tension": 0.05}
                    },
                }
            },
            "プレイヤーの部屋": {
                "slots": {
                    "night": {
                        "emotions": {"affection": 0.25, "arousal": 0.10, "tension": -0.10}
                    }
                }
            },
            "プール": {
                "slots": {
                    "after_school": {
                        "emotions": {"affection": 0.30, "arousal": 0.20, "tension": 0.10}
                    }
                }
            },
        }

    # ====== ランタイム用 ======
    def _parse_time(self, hhmm: str) -> Optional[time]:
        try:
            return datetime.strptime(hhmm, "%H:%M").time()
        except Exception:
            return None

    def _find_slot_for_time(self, current: time) -> Optional[str]:
        """現在時刻にマッチする time_slot を返す。なければ None。"""
        for slot_name, spec in self.time_slots.items():
            t_start = self._parse_time(spec.get("start", "00:00"))
            t_end = self._parse_time(spec.get("end", "23:59"))
            if not t_start or not t_end:
                continue
            if t_start <= current < t_end:
                return slot_name
        return None

    def get_for(
        self,
        location: str,
        *,
        time_str: Optional[str] = None,
        slot_name: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        指定された場所 + 時刻/スロットに対応する感情ベクトルを返す。

        - slot_name を明示指定 → そのスロットの emotions
        - time_str="HH:MM" が渡された場合 → time_slots から該当スロットを探索
        - 見つからない場合は 0 ベクトル（全次元 0.0）
        """
        # スロット確定
        if slot_name is None and time_str:
            t = self._parse_time(time_str)
            if t:
                slot_name = self._find_slot_for_time(t)

        if slot_name is None:
            # フォールバック：最初のスロット
            slot_name = next(iter(self.time_slots.keys()), None)

        if slot_name is None:
            return {dim: 0.0 for dim in self.dimensions}

        loc = self.locations.get(location, {})
        slots = loc.get("slots", {})
        emo = slots.get(slot_name, {}).get("emotions", {})

        return {dim: float(emo.get(dim, 0.0)) for dim in self.dimensions}

    # ====== Streamlit UI ======
    def render(self) -> None:
        """SceneManager エディタ UI。"""
        st.markdown("## 🌏 Scene Emotion Manager")
        st.caption(f"保存先: `{self.path}`")

        if not self.time_slots or not self.locations:
            self._init_default()

        # ---- 時間帯スロット編集 ----
        st.markdown("### ⏱ 時間帯スロット設定")

        for name in list(self.time_slots.keys()):
            spec = self.time_slots.setdefault(name, {"start": "00:00", "end": "23:59"})
            col1, col2, col3 = st.columns([1.2, 1, 1])
            with col1:
                st.markdown(f"**{name}**")
            with col2:
                spec["start"] = st.text_input(
                    f"{name} / start (HH:MM)",
                    value=spec.get("start", "00:00"),
                    key=f"ts_{name}_start",
                )
            with col3:
                spec["end"] = st.text_input(
                    f"{name} / end (HH:MM)",
                    value=spec.get("end", "23:59"),
                    key=f"ts_{name}_end",
                )

        with st.expander("➕ 時間帯スロット追加", expanded=False):
            new_name = st.text_input("新しい時間帯名（例: evening）", key="ts_new_name")
            if st.button("時間帯を追加", key="ts_add_btn"):
                name = new_name.strip()
                if name:
                    if name in self.time_slots:
                        st.warning(f"時間帯『{name}』は既に存在します。")
                    else:
                        self.time_slots[name] = {"start": "00:00", "end": "23:59"}
                        st.success(f"時間帯『{name}』を追加しました。")
                        st.experimental_rerun()

        st.markdown("---")

        # ---- ロケーション別 一日スケジュール ----
        st.markdown("### 🏙 ロケーション別・一日スケジュール")

        for loc_name in list(self.locations.keys()):
            loc = self.locations.setdefault(loc_name, {"slots": {}})
            slots = loc.setdefault("slots", {})

            with st.expander(f"📍 {loc_name}", expanded=True):
                for slot_name, ts_spec in self.time_slots.items():
                    emo = slots.setdefault(slot_name, {"emotions": {}})
                    emo_vec = emo.setdefault("emotions", emo.get("emotions", {}))

                    label = f"{slot_name} ({ts_spec.get('start')}–{ts_spec.get('end')})"
                    st.markdown(f"**{label}**")

                    # 感情次元ごとのスライダー
                    cols = st.columns(len(self.dimensions))
                    for i, dim in enumerate(self.dimensions):
                        with cols[i]:
                            default_val = float(emo_vec.get(dim, 0.0))
                            emo_vec[dim] = st.slider(
                                f"{loc_name}/{slot_name}/{dim}",
                                -1.0,
                                1.0,
                                default_val,
                                0.05,
                                key=f"loc_{loc_name}_{slot_name}_{dim}",
                            )
                st.markdown("---")

        with st.expander("➕ 場所を追加", expanded=False):
            new_loc = st.text_input("新しい場所名（例: 屋上）", key="loc_new_name")
            if st.button("場所を追加", key="loc_add_btn"):
                name = new_loc.strip()
                if name:
                    if name in self.locations:
                        st.warning(f"場所『{name}』は既に存在します。")
                    else:
                        self.locations[name] = {"slots": {}}
                        st.success(f"場所『{name}』を追加しました。")
                        st.experimental_rerun()

        # ---- 保存 ----
        if st.button("💾 保存", type="primary", key="scene_save_btn"):
            self.save()
            st.success("Scene 情報を保存しました。")

        # ---- デバッグプレビュー ----
        with st.expander("🧪 JSON preview", expanded=False):
            st.json(
                {
                    "dimensions": self.dimensions,
                    "time_slots": self.time_slots,
                    "locations": self.locations,
                }
            )
