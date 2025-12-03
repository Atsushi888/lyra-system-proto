# actors/scene/scene_manager.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime, time
import json
import os

import streamlit as st


# デフォルトで持つ感情ディメンション
DEFAULT_DIMENSIONS: List[str] = [
    "affection",   # 好意
    "arousal",     # 興奮（性的/情動）
    "tension",     # 緊張
    "anger",       # 怒り
    "sadness",     # 悲しみ
    "excitement",  # 期待・ワクワク
]

# 日本語ラベル
DIM_JA_LABELS: Dict[str, str] = {
    "affection":  "affection（好意）",
    "arousal":    "arousal（興奮・性的/情動）",
    "tension":    "tension（緊張）",
    "anger":      "anger（怒り）",
    "sadness":    "sadness（悲しみ）",
    "excitement": "excitement（期待・ワクワク）",
}


@dataclass
class SceneManager:
    """
    場所ごとに「一日の時間帯スロット」と「感情補正ベクトル」を持つマネージャ。

    JSON構造（v2.0-slot の例）:
    {
      "meta": {
        "version": "2.0-slot",
        "updated_at": "...",
        "dimensions": ["affection", "arousal", ...]
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

    # JSON 保存先
    path: str = "actors/scene/scene_bonus/scene_emotion_map.json"

    # 感情次元（UI はこのリストに従ってスライダーを出す）
    dimensions: List[str] = field(
        default_factory=lambda: list(DEFAULT_DIMENSIONS)
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

        loaded_dims = meta.get("dimensions") or []
        # 既存ファイルに足りないデフォルト次元があれば足す
        dims: List[str] = []
        for d in loaded_dims:
            if d not in dims:
                dims.append(d)
        for d in DEFAULT_DIMENSIONS:
            if d not in dims:
                dims.append(d)
        self.dimensions = dims

        self.time_slots = data.get("time_slots", {})
        self.locations = data.get("locations", {})

        if not self.time_slots or not self.locations:
            self._init_default()
            return

        # 新しく追加されたディメンションを全ロケーションへ 0.0 で埋める
        for d in self.dimensions:
            self._ensure_dimension_exists_everywhere(d)

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
        self.dimensions = list(DEFAULT_DIMENSIONS)

        self.time_slots = {
            "morning":      {"start": "07:00", "end": "09:00"},
            "lunch":        {"start": "12:00", "end": "13:00"},
            "after_school": {"start": "16:00", "end": "19:00"},
            "night":        {"start": "20:00", "end": "23:30"},
        }

        # 6軸そろえたゼロベクトル
        base_zeros = {dim: 0.0 for dim in self.dimensions}

        def vec(**kwargs: float) -> Dict[str, float]:
            v = base_zeros.copy()
            v.update(kwargs)
            return v

        self.locations = {
            "通学路": {
                "slots": {
                    "morning": {
                        "emotions": vec(
                            affection=0.10,
                            arousal=-0.10,
                            tension=-0.10,
                        )
                    },
                    "after_school": {
                        "emotions": vec(
                            affection=0.25,
                            arousal=0.20,
                            tension=0.10,
                        )
                    },
                }
            },
            "学食": {
                "slots": {
                    "lunch": {
                        "emotions": vec(
                            affection=0.20,
                            arousal=-0.20,
                            tension=-0.10,
                        )
                    }
                }
            },
            "駅前": {
                "slots": {
                    "after_school": {
                        "emotions": vec(
                            affection=0.15,
                            arousal=0.00,
                            tension=0.00,
                        )
                    },
                    "night": {
                        "emotions": vec(
                            affection=0.18,
                            arousal=0.10,
                            tension=0.05,
                        )
                    },
                }
            },
            "プレイヤーの部屋": {
                "slots": {
                    "night": {
                        "emotions": vec(
                            affection=0.25,
                            arousal=0.10,
                            tension=-0.10,
                        )
                    },
                    "morning": {
                        "emotions": vec(
                            affection=0.20,
                            arousal=0.05,
                            tension=-0.05,
                        )
                    },
                }
            },
            "プール": {
                "slots": {
                    "after_school": {
                        "emotions": vec(
                            affection=0.30,
                            arousal=0.20,
                            tension=0.10,
                        )
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

    # ====== ユーティリティ ======
    def _ensure_dimension_exists_everywhere(self, dim: str) -> None:
        """新しい感情次元を、全ロケーション・全スロットに 0.0 で追加する。"""
        for loc in self.locations.values():
            slots = loc.setdefault("slots", {})
            for slot in slots.values():
                emo = slot.setdefault("emotions", {})
                emo.setdefault(dim, 0.0)

    def _dim_label(self, dim: str) -> str:
        """UI 表示用ラベル（日本語訳つき）。"""
        return DIM_JA_LABELS.get(dim, dim)

    # ====== Streamlit UI ======
    def render(self) -> None:
        """SceneManager エディタ UI。"""
        st.markdown("## 🌏 Scene Emotion Manager")
        st.caption(f"保存先: `{self.path}`")

        if not self.time_slots or not self.locations:
            self._init_default()

        loc_names = list(self.locations.keys())
        if not loc_names:
            st.warning("場所が定義されていません。下部の『場所を追加』から作成してください。")
            return

        slot_keys = list(self.time_slots.keys())

        # === 0) コミット済み world_state のデフォルトを保証 ===
        s = st.session_state

        if "scene_location" not in s:
            if "プレイヤーの部屋" in self.locations:
                s["scene_location"] = "プレイヤーの部屋"
            else:
                s["scene_location"] = loc_names[0]

        if "scene_time_slot" not in s:
            if "morning" in self.time_slots:
                s["scene_time_slot"] = "morning"
            elif slot_keys:
                s["scene_time_slot"] = slot_keys[0]
            else:
                s["scene_time_slot"] = None

        if "scene_time_str" not in s:
            # デフォルトは time_slot の start、なければ 07:30
            slot_name = s.get("scene_time_slot")
            default_time = "07:30"
            if slot_name and slot_name in self.time_slots:
                default_time = self.time_slots[slot_name].get("start", default_time)
            s["scene_time_str"] = default_time

        current_loc = s.get("scene_location")
        current_slot = s.get("scene_time_slot")
        current_time_str = s.get("scene_time_str")

        # === ① 現在の world_state 表示 ===
        st.markdown("### 🎯 現在の world_state & 感情補正")

        col_a, col_b, col_c = st.columns([1.5, 1.2, 1.2])
        with col_a:
            st.write(f"**場所**: {current_loc}")
        with col_b:
            if current_slot:
                spec = self.time_slots.get(current_slot, {})
                st.write(
                    f"**時間帯スロット**: {current_slot} "
                    f"({spec.get('start', '--:--')}–{spec.get('end', '--:--')})"
                )
            else:
                st.write("**時間帯スロット**: 自動判定")
        with col_c:
            st.write(f"**時刻**: {current_time_str or '（未設定）'}")

        # 現在地の感情ベクトル
        current_emo = self.get_for(
            location=current_loc,
            time_str=current_time_str,
            slot_name=current_slot,
        )

        with st.expander("現在の world_state → scene_emotion", expanded=True):
            for dim in self.dimensions:
                label = self._dim_label(dim)
                val = float(current_emo.get(dim, 0.0))
                st.write(f"- {label}: {val:+.2f}")

        st.markdown("---")

        # === ② 移動プラン設定 ===
        st.markdown("### 🚶‍♀️ プレイヤー移動プラン")

        # プラン用 state（未設定なら現在値で初期化）
        plan_loc = s.get("scene_plan_location", current_loc)
        plan_slot = s.get("scene_plan_time_slot", current_slot or "auto")
        plan_time = s.get("scene_plan_time_str", current_time_str or "")

        slot_label_auto = "auto（時刻から判定）"
        slot_options = [slot_label_auto] + slot_keys

        # 入力 UI
        col1, col2, col3 = st.columns([1.5, 1.2, 1.2])
        with col1:
            plan_loc = st.selectbox(
                "プレイヤーの移動先",
                options=loc_names,
                index=loc_names.index(plan_loc) if plan_loc in loc_names else 0,
                key="sm_plan_loc",
            )
        with col2:
            initial_slot = plan_slot if plan_slot in slot_keys else slot_label_auto
            plan_slot = st.selectbox(
                "移動先の時間帯スロット",
                options=slot_options,
                index=slot_options.index(initial_slot),
                key="sm_plan_slot",
            )
        with col3:
            plan_time = st.text_input(
                "移動先の時刻（HH:MM）",
                value=plan_time or "",
                key="sm_plan_time_str",
            ).strip()

        # プレビューベクトル
        preview_slot: Optional[str] = None if plan_slot == slot_label_auto else plan_slot
        preview_time: Optional[str] = plan_time or None
        preview_emo = self.get_for(
            location=plan_loc,
            time_str=preview_time,
            slot_name=preview_slot,
        )

        with st.expander("移動先 world_state プレビュー", expanded=False):
            st.write(f"**場所**: {plan_loc}")
            if preview_slot:
                spec = self.time_slots.get(preview_slot, {})
                st.write(
                    f"**時間帯スロット**: {preview_slot} "
                    f"({spec.get('start', '--:--')}–{spec.get('end', '--:--')})"
                )
            else:
                st.write("**時間帯スロット**: auto（時刻から判定）")
            st.write(f"**時刻**: {preview_time or '（未設定）'}")
            st.markdown("**感情補正ベクトル:**")
            for dim in self.dimensions:
                label = self._dim_label(dim)
                val = float(preview_emo.get(dim, 0.0))
                st.write(f"- {label}: {val:+.2f}")

        # 移動ボタン
        if st.button("🚕 この設定で移動する", type="primary", key="sm_do_move"):
            # state にコミット
            s["scene_location"] = plan_loc
            s["scene_time_slot"] = None if plan_slot == slot_label_auto else plan_slot
            s["scene_time_str"] = plan_time or None

            # プラン state も現在値として覚えておく
            s["scene_plan_location"] = plan_loc
            s["scene_plan_time_slot"] = plan_slot
            s["scene_plan_time_str"] = plan_time

            # llm_meta に world_state を書き込む（存在すれば）
            ws = {
                "location": plan_loc,
                "time_slot": s["scene_time_slot"],
                "time_str": s["scene_time_str"],
            }
            llm_meta = s.get("llm_meta")
            if isinstance(llm_meta, dict):
                llm_meta["world_state"] = ws
                s["llm_meta"] = llm_meta

            # CouncilManager へ「場所が変わったよ」と通知
            s["world_state_changed"] = True

            st.success("プレイヤーの場所と時刻を更新しました。")
            st.rerun()

        st.markdown("---")

        # ---- ③ 時間帯スロット編集 ----
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
                        st.rerun()

        st.markdown("---")

        # ---- ④ 感情ディメンション ----
        st.markdown("### 🎭 感情ディメンション")

        # 日本語ラベル付きで表示
        disp_dims = [self._dim_label(d) for d in self.dimensions]
        st.write("現在の次元:", ", ".join(disp_dims))

        with st.expander("➕ 感情ディメンションを追加", expanded=False):
            new_dim = st.text_input(
                "新しい感情名（例: comfort / loneliness）",
                key="dim_new_name",
            )
            if st.button("ディメンション追加", key="dim_add_btn"):
                name = new_dim.strip()
                if name:
                    if name in self.dimensions:
                        st.warning(f"感情次元『{name}』は既に存在します。")
                    else:
                        self.dimensions.append(name)
                        self._ensure_dimension_exists_everywhere(name)
                        st.success(f"感情次元『{name}』を追加しました。")
                        st.rerun()

        st.markdown("---")

        # ---- ⑤ ロケーション別 一日スケジュール ----
        st.markdown("### 🏙 ロケーション別・一日スケジュール")

        max_per_row = 3  # スライダー 3 本ごとに改行

        for loc_name in list(self.locations.keys()):
            loc = self.locations.setdefault(loc_name, {"slots": {}})
            slots = loc.setdefault("slots", {})

            with st.expander(f"📍 {loc_name}", expanded=True):
                for slot_name, ts_spec in self.time_slots.items():
                    emo = slots.setdefault(slot_name, {"emotions": {}})
                    emo_vec = emo.setdefault("emotions", emo.get("emotions", {}))

                    label = f"{slot_name} ({ts_spec.get('start')}–{ts_spec.get('end')})"
                    st.markdown(f"**{label}**")

                    # 感情ディメンションを max_per_row ごとに折り返す
                    dims = list(self.dimensions)
                    for i in range(0, len(dims), max_per_row):
                        chunk = dims[i: i + max_per_row]
                        cols = st.columns(len(chunk))
                        for dim, col in zip(chunk, cols):
                            with col:
                                default_val = float(emo_vec.get(dim, 0.0))
                                emo_vec[dim] = st.slider(
                                    f"{loc_name}/{slot_name}/{dim}",
                                    -1.0,
                                    1.0,
                                    default_val,
                                    0.05,
                                    key=f"loc_{loc_name}_{slot_name}_{dim}",
                                    help=self._dim_label(dim),
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
                        # 全スロットに 0.0 で初期化する
                        slots: Dict[str, Any] = {}
                        for slot_name in self.time_slots.keys():
                            slots[slot_name] = {
                                "emotions": {
                                    dim: 0.0 for dim in self.dimensions
                                }
                            }
                        self.locations[name] = {"slots": slots}
                        st.success(f"場所『{name}』を追加しました。")
                        st.rerun()

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
