# components/ai_manager.py
from __future__ import annotations

from typing import Any, Dict, List
import streamlit as st

from llm.llm_manager import LLMManager


class AIManager:
    """
    AI選択・優先順位・警告抑制・(ついでに)プレイヤー名変更をまとめるUI。
    """

    TITLE = "🤖 AI Manager"

    # 初期優先順位（最小構成：gpt52のみ）
    DEFAULT_PRIORITY = ["gpt52"]

    def __init__(self, persona_id: str = "default") -> None:
        self.persona_id = persona_id
        self.llm_manager = LLMManager.get_or_create(persona_id=persona_id)

        if "ai_manager" not in st.session_state or not isinstance(st.session_state["ai_manager"], dict):
            st.session_state["ai_manager"] = {}

        self.state: Dict[str, Any] = st.session_state["ai_manager"]
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        st.session_state.setdefault("player_name", "アツシ")

        self.state.setdefault("x_rated", False)
        self.state.setdefault("suppress_warnings", False)

        # ★初期は Manual
        self.state.setdefault("select_mode", "Manual")  # "Auto" or "Manual"

        st.session_state.setdefault("reply_length_mode", "auto")

        props = self.llm_manager.get_model_props() or {}

        # ★初期は gpt52 だけ True（propsにあるものだけ）
        if "enabled_models" not in self.state or not isinstance(self.state["enabled_models"], dict):
            enabled_map: Dict[str, bool] = {name: False for name in props.keys()}
            if "gpt52" in enabled_map:
                enabled_map["gpt52"] = True
            self.state["enabled_models"] = enabled_map
        else:
            # 欠けモデルを補完
            enabled_map = self.state["enabled_models"]
            for name in props.keys():
                enabled_map.setdefault(name, False)
            if "gpt52" in enabled_map and enabled_map.get("gpt52") is None:
                enabled_map["gpt52"] = True

        # priority list
        if "priority" not in self.state or not isinstance(self.state["priority"], list):
            available = list(props.keys())
            pri: List[str] = []
            for x in self.DEFAULT_PRIORITY:
                if x in available and x not in pri:
                    pri.append(x)
            for x in available:
                if x not in pri:
                    pri.append(x)
            self.state["priority"] = pri

    def _apply_enabled_to_manager(self) -> None:
        enabled = self.state.get("enabled_models") or {}
        if isinstance(enabled, dict):
            self.llm_manager.set_enabled_models(enabled)

    def _ordered_models(self, props: Dict[str, Dict[str, Any]]) -> List[str]:
        priority = self.state.get("priority") or []
        if not isinstance(priority, list):
            priority = []

        existing = set(props.keys())
        ordered: List[str] = [m for m in priority if m in existing]
        for m in props.keys():
            if m not in ordered:
                ordered.append(m)
        return ordered

    def render(self) -> None:
        st.header(self.TITLE)

        with st.expander("🧑 プレイヤー名（Persona へ渡す）", expanded=True):
            cur_name = st.session_state.get("player_name", "アツシ")
            new_name = st.text_input("player_name", value=str(cur_name), key="ai_mgr_player_name_input")

            cols = st.columns(2)
            with cols[0]:
                if st.button("適用", use_container_width=True):
                    st.session_state["player_name"] = str(new_name).strip() or "アツシ"
                    st.success(f"player_name を `{st.session_state['player_name']}` に更新しました。")
                    st.rerun()
            with cols[1]:
                st.caption("※ Persona は View 側で player_name を受け取り、{PLAYER_NAME} を置換します。")

        with st.expander("⚙️ 動作モード", expanded=True):
            self.state["select_mode"] = st.radio(
                "AI 選択モード",
                options=["Auto", "Manual"],
                index=0 if self.state.get("select_mode", "Manual") == "Auto" else 1,
                horizontal=True,
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                self.state["x_rated"] = st.checkbox(
                    "X-Rated",
                    value=bool(self.state.get("x_rated", False)),
                )
            with c2:
                self.state["suppress_warnings"] = st.checkbox(
                    "警告抑制（suppress_warnings）",
                    value=bool(self.state.get("suppress_warnings", False)),
                    help="回答が取れないAIがあっても st.error 等で騒がないためのスイッチ。",
                )
            with c3:
                st.selectbox(
                    "発話長さモード（reply_length_mode）",
                    options=["auto", "short", "normal", "long", "story"],
                    index=["auto", "short", "normal", "long", "story"].index(
                        str(st.session_state.get("reply_length_mode", "auto") or "auto")
                    ),
                    key="reply_length_mode",
                )

        props = self.llm_manager.get_model_props() or {}
        ordered = self._ordered_models(props)

        st.subheader("📋 利用可能モデル（有効/無効と優先順位）")

        if not props:
            st.warning("モデル情報が取得できませんでした（get_model_props が空）。")
            return

        with st.expander("🧭 優先順位（priority）", expanded=True):
            st.caption("上から順に優先。")

            current_priority: List[str] = list(self.state.get("priority") or [])
            base_list = [m for m in current_priority if m in props]
            for m in ordered:
                if m not in base_list:
                    base_list.append(m)

            new_priority: List[str] = []
            remaining = base_list[:]
            for i in range(len(base_list)):
                choice = st.selectbox(
                    f"優先 {i+1}",
                    options=remaining,
                    index=0,
                    key=f"ai_mgr_priority_{i}",
                )
                new_priority.append(choice)
                if choice in remaining:
                    remaining.remove(choice)
                if not remaining:
                    break

            cols = st.columns(2)
            with cols[0]:
                if st.button("優先順位を保存", use_container_width=True):
                    uniq: List[str] = []
                    for m in new_priority:
                        if m in props and m not in uniq:
                            uniq.append(m)
                    for m in props.keys():
                        if m not in uniq:
                            uniq.append(m)
                    self.state["priority"] = uniq
                    st.success("priority を保存しました。")
                    st.rerun()
            with cols[1]:
                st.caption(f"現在: {', '.join(self.state.get('priority') or [])}")

        with st.expander("✅ モデルの有効/無効（enabled）", expanded=True):
            enabled_map = self.state.get("enabled_models") or {}
            if not isinstance(enabled_map, dict):
                enabled_map = {}
                self.state["enabled_models"] = enabled_map

            for name in ordered:
                p = props.get(name, {}) or {}
                default_on = bool(p.get("enabled", True))
                current_on = bool(enabled_map.get(name, default_on))
                enabled_map[name] = st.checkbox(
                    f"{name}",
                    value=current_on,
                    key=f"ai_mgr_enabled_{name}",
                    help=str(p.get("label") or p.get("provider") or ""),
                )

            cols = st.columns(2)
            with cols[0]:
                if st.button("enabled を反映", use_container_width=True):
                    self._apply_enabled_to_manager()
                    st.success("LLMManager に enabled 設定を反映しました。")
                    st.rerun()
            with cols[1]:
                st.caption("※ UI表示だけでなく、LLMManager 側の enabled にも反映します。")

        st.subheader("🧾 現在の設定サマリ")
        st.json(
            {
                "player_name": st.session_state.get("player_name"),
                "reply_length_mode": st.session_state.get("reply_length_mode"),
                "select_mode": self.state.get("select_mode"),
                "x_rated": self.state.get("x_rated"),
                "suppress_warnings": self.state.get("suppress_warnings"),
                "priority": self.state.get("priority"),
                "enabled_models": self.state.get("enabled_models"),
            }
        )
