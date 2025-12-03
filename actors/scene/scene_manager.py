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
                    }
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

        # === ① 冒頭：プレイヤー所在地 & 現在時刻テスト ===
        st.markdown("### 🎯 プレイヤー所在地 & 現在時刻テスト")

        loc_names = list(self.locations.keys())
        if not loc_names:
            st.info("場所がまだ定義されていません。下のエディタで追加してください。")
        else:
            # 既存 state からデフォルトを拾う
            default_loc = st.session_state.get("scene_location", loc_names[0])
            if default_loc not in loc_names:
                default_loc = loc_names[0]

            col_top1, col_top2 = st.columns([2, 1])

            with col_top1:
                selected_loc = st.selectbox(
                    "プレイヤーの現在地",
                    options=loc_names,
                    index=loc_names.index(default_loc),
                    key="sm_world_loc",
                )

            slot_keys = list(self.time_slots.keys())
            slot_label_auto = "（自動判定：時刻から決定）"
            slot_options = [slot_label_auto] + slot_keys

            with col_top2:
                default_slot = st.session_state.get("scene_time_slot")
                if default_slot not in slot_keys:
                    default_slot = slot_label_auto
                selected_slot = st.selectbox(
                    "時間帯スロット（任意）",
                    options=slot_options,
                    index=slot_options.index(default_slot)
                    if default_slot in slot_options
                    else 0,
                    key="sm_world_slot",
                )

            col_time, col_dummy = st.columns([1.2, 2])
            with col_time:
                default_time_str = st.session_state.get("scene_time_str", "07:30")
                time_str = st.text_input(
                    "現在時刻（HH:MM）※空ならスロットのみで判定",
                    value=default_time_str,
                    key="sm_world_time_str",
                ).strip()

            # スロット名決定
            slot_name: Optional[str]
            if selected_slot == slot_label_auto:
                slot_name = None
            else:
                slot_name = selected_slot

            time_str_clean: Optional[str] = time_str or None

            # SceneManager から感情ベクトル取得
            emo_vec = self.get_for(
                location=selected_loc,
                time_str=time_str_clean,
                slot_name=slot_name,
            )

            # → SceneAI 側と共有したい world_state を session_state に書き込む
            st.session_state["scene_location"] = selected_loc
            if slot_name is not None:
                st.session_state["scene_time_slot"] = slot_name
            if time_str_clean is not None:
                st.session_state["scene_time_str"] = time_str_clean

            # ★ world_state が変わったら CouncilManager をリセットさせる
            world_key = f"{selected_loc}|{slot_name or ''}|{time_str_clean or ''}"
            prev_key = st.session_state.get("scene_world_state_key")
            if world_key != prev_key:
                st.session_state["scene_world_state_key"] = world_key
                # 会談マネージャを作り直させる（Round0 を新しい場所で生成させる）
                if "council_manager" in st.session_state:
                    st.session_state.pop("council_manager")

            # 結果表示
            with st.expander("現在の world_state → scene_emotion", expanded=True):
                st.write(f"**場所**: {selected_loc}")
                if slot_name:
                    spec = self.time_slots.get(slot_name, {})
                    st.write(
                        f"**時間帯スロット**: {slot_name} "
                        f"({spec.get('start', '--:--')}–{spec.get('end', '--:--')})"
                    )
                else:
                    st.write("**時間帯スロット**: 時刻から自動判定")
                st.write(f"**時刻文字列**: {time_str_clean or '（未指定）'}")

                st.markdown("**感情補正ベクトル:**")
                for dim in self.dimensions:
                    label = self._dim_label(dim)
                    val = float(emo_vec.get(dim, 0.0))
                    st.write(f"- {label}: {val:+.2f}")

        st.markdown("---")

        # ---- ② 時間帯スロット編集 ----
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

        # ---- ③ 感情ディメンション ----
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

        # ---- ④ ロケーション別 一日スケジュール ----
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
                        chunk = dims[i : i + max_per_row]
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
