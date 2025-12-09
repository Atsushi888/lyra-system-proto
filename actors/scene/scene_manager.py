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


# ==========================================================
# ★ 追加: 相手キャラ名の取得 & Council リセットユーティリティ
# ==========================================================
def _get_partner_display_name() -> str:
    """
    相手として設定されている Persona から名前を参照。
    取得できなければ『リセリア』を返す。
    """
    default_name = "リセリア"

    try:
        llm_meta = st.session_state.get("llm_meta") or {}
        persona = llm_meta.get("persona") or {}
        profile = persona.get("profile") or {}
        name = (
            profile.get("public_name")
            or persona.get("display_name")
            or default_name
        )
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception:
        pass

    return default_name


def _reset_council_state(world_before: Dict[str, Any],
                         world_after: Dict[str, Any]) -> None:
    """
    world_state の「プレイヤー/相手の場所・時間」に変化があれば、
    Council 系の状態を Round0 相当にリセットする。
    """
    def _extract(ws: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(ws, dict):
            ws = {}
        loc = ws.get("locations") or {}
        if not isinstance(loc, dict):
            loc = {}
        t = ws.get("time") or {}
        if not isinstance(t, dict):
            t = {}
        return {
            "player": loc.get("player"),
            "partner": loc.get("floria"),  # world_state 内キーは従来どおり "floria"
            "slot": t.get("slot"),
            "time_str": t.get("time_str"),
        }

    before_core = _extract(world_before)
    after_core = _extract(world_after)

    if before_core == after_core:
        # 場所・時間の何も変化がなければ何もしない
        return

    # CouncilManager インスタンスがあれば reset() を呼ぶ
    mgr = st.session_state.get("council_manager")
    if mgr is not None and hasattr(mgr, "reset"):
        try:
            mgr.reset()
        except Exception:
            pass

    # 汎用的なセッションキーも初期化
    st.session_state["council_history"] = []
    st.session_state["council_round"] = 0

    try:
        st.toast("場所／時間の変更を検知 → Council 表示をリセットしました。")
    except Exception:
        st.info("場所／時間の変更を検知 → Council 表示をリセットしました。")


@dataclass
class SceneManager:
    """
    場所ごとに「一日の時間帯スロット」と「感情補正ベクトル」を持つマネージャ。
    """

    # JSON 保存先
    path: str = "actors/scene/scene_bonus/scene_emotion_map.json"

    # 感情次元
    dimensions: List[str] = field(
        default_factory=lambda: list(DEFAULT_DIMENSIONS)
    )

    # "morning" → {"start": "07:00", "end": "09:00"}
    time_slots: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # "通学路" → {"slots": { "morning": {"emotions": {...}}, ... }}
    locations: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ====== 基本 I/O ======
    def load(self) -> None:
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

        if version != "2.0-slot":
            self._init_default()
            return

        loaded_dims = meta.get("dimensions") or []
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

        for d in self.dimensions:
            self._ensure_dimension_exists_everywhere(d)

    def save(self) -> None:
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
        self.dimensions = list(DEFAULT_DIMENSIONS)

        self.time_slots = {
            "morning":      {"start": "07:00", "end": "09:00"},
            "lunch":        {"start": "12:00", "end": "13:00"},
            "after_school": {"start": "16:00", "end": "19:00"},
            "night":        {"start": "20:00", "end": "23:30"},
        }

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
                            affection=0.20,
                            arousal=-0.10,
                            tension=0.10,
                            excitement=0.20,
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
                            excitement=0.10,
                        )
                    }
                }
            },
            "駅前": {
                "slots": {
                    "morning": {
                        "emotions": vec(
                            affection=0.05,
                            arousal=0.00,
                            tension=0.05,
                            anger=0.05,
                            excitement=0.10,
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
                    "morning": {
                        "emotions": vec(
                            affection=0.15,
                            arousal=0.00,
                            tension=-0.05,
                        )
                    },
                    "night": {
                        "emotions": vec(
                            affection=0.25,
                            arousal=0.10,
                            tension=-0.10,
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
        if slot_name is None and time_str:
            t = self._parse_time(time_str)
            if t:
                slot_name = self._find_slot_for_time(t)

        if slot_name is None:
            slot_name = next(iter(self.time_slots.keys()), None)

        if slot_name is None:
            return {dim: 0.0 for dim in self.dimensions}

        loc = self.locations.get(location, {})
        slots = loc.get("slots", {})
        emo = slots.get(slot_name, {}).get("emotions", {})

        return {dim: float(emo.get(dim, 0.0)) for dim in self.dimensions}

    # ====== ユーティリティ ======
    def _ensure_dimension_exists_everywhere(self, dim: str) -> None:
        for loc in self.locations.values():
            slots = loc.setdefault("slots", {})
            for slot in slots.values():
                emo = slot.setdefault("emotions", {})
                emo.setdefault(dim, 0.0)

    def _dim_label(self, dim: str) -> str:
        return DIM_JA_LABELS.get(dim, dim)

    # ====== Streamlit UI ======
    def render(self) -> None:
        from actors.scene_ai import SceneAI  # 循環参照回避のためローカル import

        st.markdown("## 🌏 Scene Emotion Manager")
        st.caption(f"保存先: `{self.path}`")

        if not self.time_slots or not self.locations:
            self._init_default()

        # ★ 相手キャラ表示名（Persona → 取れなければ「リセリア」）
        partner_name = _get_partner_display_name()

        scene_ai = SceneAI(state=st.session_state)
        world = scene_ai.get_world_state()
        locs = world.get("locations", {})
        t = world.get("time", {})
        party = world.get("party", {})

        player_loc = locs.get("player", "プレイヤーの部屋")
        floria_loc = locs.get("floria", "プレイヤーの部屋")
        current_slot = t.get("slot", "morning")
        current_time_str = t.get("time_str", "07:30")
        party_mode = party.get("mode", "with_floria")

        # 現在の world_state に基づく感情補正（プレイヤー位置）
        current_emo = self.get_for(
            location=player_loc,
            time_str=current_time_str,
            slot_name=current_slot,
        )

        # === ① 現在の world_state 表示 ===
        st.markdown("### 🎯 現在の world_state & 感情補正")

        cols = st.columns([2, 2, 1])
        with cols[0]:
            st.write(f"プレイヤー: **{player_loc}**")
            st.write(f"{partner_name}: **{floria_loc}**")
        with cols[1]:
            slot_spec = self.time_slots.get(current_slot, {})
            st.write(
                f"時間帯スロット: **{current_slot}** "
                f"({slot_spec.get('start', '--:--')}–{slot_spec.get('end', '--:--')})"
            )
            st.write(f"パーティ状態: **{party_mode}**")
        with cols[2]:
            st.write(f"時刻: **{current_time_str}**")

        with st.expander("現在の world_state → scene_emotion（プレイヤー位置）", expanded=True):
            st.markdown("**感情補正ベクトル:**")
            for dim in self.dimensions:
                label = self._dim_label(dim)
                val = float(current_emo.get(dim, 0.0))
                st.write(f"- {label}: {val:+.2f}")

        st.markdown("---")

        # === ② プレイヤー移動プラン ===
        st.markdown("### 🚶‍♀️ プレイヤー移動プラン")

        slot_keys = list(self.time_slots.keys())
        if current_slot not in slot_keys and slot_keys:
            current_slot = slot_keys[0]

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            dest_loc = st.selectbox(
                "プレイヤーの移動先",
                options=list(self.locations.keys()),
                index=list(self.locations.keys()).index(player_loc)
                if player_loc in self.locations
                else 0,
                key="sm_move_dest_loc_player",
            )
        with col2:
            dest_slot = st.selectbox(
                "移動先の時間帯スロット",
                options=slot_keys,
                index=slot_keys.index(current_slot) if current_slot in slot_keys else 0,
                key="sm_move_slot_player",
            )
        with col3:
            dest_time_str = st.text_input(
                "移動先の時刻（HH:MM）",
                value=current_time_str,
                key="sm_move_time_str_player",
            ).strip() or current_time_str

        # プレビュー（プレイヤー）
        dest_emo = self.get_for(
            location=dest_loc,
            time_str=dest_time_str,
            slot_name=dest_slot,
        )

        with st.expander("移動先 world_state プレビュー（プレイヤー）", expanded=False):
            spec = self.time_slots.get(dest_slot, {})
            st.write(f"場所: **{dest_loc}**")
            st.write(
                f"時間帯スロット: **{dest_slot}** "
                f"({spec.get('start', '--:--')}–{spec.get('end', '--:--')})"
            )
            st.write(f"時刻: **{dest_time_str}**")
            st.markdown("**感情補正ベクトル（移動先）:**")
            for dim in self.dimensions:
                label = self._dim_label(dim)
                val = float(dest_emo.get(dim, 0.0))
                st.write(f"- {label}: {val:+.2f}")

        if st.button("✨ この条件でプレイヤーを移動する",
                     type="primary",
                     key="sm_do_move_player"):
            world_before = world
            scene_ai.move_player(
                dest_loc,
                time_slot=dest_slot,
                time_str=dest_time_str,
            )
            world_after = scene_ai.get_world_state()

            _reset_council_state(world_before, world_after)

            st.success("プレイヤーを移動しました。")
            st.rerun()

        st.markdown("---")

        # === ②' 相手（リセリア）移動プラン ===
        st.markdown(f"### 🧚‍♀️ {partner_name} 移動プラン")

        # ★ プレイヤーと同様に、現在位置＆移動先をグループ外に配置
        colf1, colf2 = st.columns([2, 2])
        with colf1:
            st.write(f"現在位置: **{floria_loc}**")
        with colf2:
            dest_loc_floria = st.selectbox(
                f"{partner_name} の移動先",
                options=list(self.locations.keys()),
                index=list(self.locations.keys()).index(floria_loc)
                if floria_loc in self.locations
                else 0,
                key="sm_move_dest_loc_floria",
            )

        # （必要なら、今後ここにプレビュー用の expander を追加してもOK）

        label_move_partner = f"✨ この条件で{partner_name}を移動する"
        if st.button(label_move_partner, key="sm_do_move_floria"):
            world_before = world
            scene_ai.move_floria(dest_loc_floria)
            world_after = scene_ai.get_world_state()

            _reset_council_state(world_before, world_after)

            st.success(f"{partner_name} の現在地を更新しました。")
            st.rerun()

        st.markdown("---")

        # === ③ 以降は従来どおり：時間帯スロット / 感情ディメンション / ロケーション別マップ編集 ===

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
                        st.rerun()

        st.markdown("---")

        # ---- 感情ディメンション ----
        st.markdown("### 🎭 感情ディメンション")

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

        # ---- ロケーション別 一日スケジュール ----
        st.markdown("### 🏙 ロケーション別・一日スケジュール")

        max_per_row = 3

        for loc_name in list(self.locations.keys()):
            loc = self.locations.setdefault(loc_name, {"slots": {}})
            slots = loc.setdefault("slots", {})

            with st.expander(f"📍 {loc_name}", expanded=True):
                for slot_name, ts_spec in self.time_slots.items():
                    emo = slots.setdefault(slot_name, {"emotions": {}})
                    emo_vec = emo.setdefault("emotions", emo.get("emotions", {}))

                    label = f"{slot_name} ({ts_spec.get('start')}–{ts_spec.get('end')})"
                    st.markdown(f"**{label}**")

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

        if st.button("💾 保存", type="primary", key="scene_save_btn"):
            self.save()
            st.success("Scene 情報を保存しました。")

        with st.expander("🧪 JSON preview", expanded=False):
            st.json(
                {
                    "dimensions": self.dimensions,
                    "time_slots": self.time_slots,
                    "locations": self.locations,
                }
            )
