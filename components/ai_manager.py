# components/ai_manager.py
from __future__ import annotations

from typing import Any, Dict, List
import streamlit as st

from llm.llm_manager import LLMManager


class AIManager:
    """
    AI選択・優先順位・警告抑制・(ついでに)プレイヤー名変更をまとめるUI。

    目的:
    - player_name をここで変更できるようにする（旧 user_settings 相当の一部）
    - 利用可能モデル一覧（props）を表示
    - enabled 切替を保存し、LLMManager に反映
    - 優先順位（順序）を session_state に保存
    - X-Rated / suppress_warnings などを session_state に保存
    """

    TITLE = "🤖 AI Manager"

    # 初期優先順位（希望: gpt52->gpt51->grok->gemini->gpt4o）
    DEFAULT_PRIORITY = ["gpt52", "gpt51", "grok", "gemini", "gpt4o"]

    def __init__(self, persona_id: str = "default") -> None:
        self.persona_id = persona_id
        self.llm_manager = LLMManager.get_or_create(persona_id=persona_id)

        # --- state slot ---
        if "ai_manager" not in st.session_state or not isinstance(st.session_state["ai_manager"], dict):
            st.session_state["ai_manager"] = {}

        self.state: Dict[str, Any] = st.session_state["ai_manager"]
        self._ensure_defaults()

    # ----------------------------
    # state defaults
    # ----------------------------
    def _ensure_defaults(self) -> None:
        # player_name はプロジェクト全体で使うのでトップレベル
        st.session_state.setdefault("player_name", "アツシ")

        # X-Rated / warn suppression
        self.state.setdefault("x_rated", False)
        self.state.setdefault("suppress_warnings", False)

        # ★初期モード：Manual（要求仕様）
        self.state.setdefault("select_mode", "Manual")  # "Auto" or "Manual"

        # reply length mode（既存キーに合わせる）
        st.session_state.setdefault("reply_length_mode", "auto")

        props = self.llm_manager.get_model_props() or {}

        # ★enabled_models：初期は gpt52 のみ True（要求仕様）
        # 既に dict があるなら尊重（上書きしない）
        if "enabled_models" not in self.state or not isinstance(self.state.get("enabled_models"), dict):
            enabled_map: Dict[str, bool] = {}
            for name in props.keys():
                enabled_map[name] = (name == "gpt52")
            self.state["enabled_models"] = enabled_map
        else:
            # 欠けているキーだけ補完
            enabled_map = self.state.get("enabled_models") or {}
            if isinstance(enabled_map, dict):
                for name in props.keys():
                    enabled_map.setdefault(name, (name == "gpt52"))

        # priority list
        if "priority" not in self.state or not isinstance(self.state["priority"], list):
            # props にあるモデルを加味して初期順序を作る
            available = list(props.keys())
            pri: List[str] = []
            for x in self.DEFAULT_PRIORITY:
                if x in available and x not in pri:
                    pri.append(x)
            for x in available:
                if x not in pri:
                    pri.append(x)
            self.state["priority"] = pri

    # ----------------------------
    # helpers
    # ----------------------------
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

    # ----------------------------
    # render
    # ----------------------------
    def render(self) -> None:
        st.header(self.TITLE)

        # ========== Player Name ==========
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

        # ========== Global switches ==========
        with st.expander("⚙️ 動作モード", expanded=True):
            # ★デフォルト Manual だが、UIは選べる
            cur_mode = self.state.get("select_mode", "Manual")
            self.state["select_mode"] = st.radio(
                "AI 選択モード",
                options=["Auto", "Manual"],
                index=0 if cur_mode == "Auto" else 1,
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
                    help="回答が取れないAIがあっても st.error 等で騒がないためのスイッチ（特にX-Rated想定）。",
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

        # ========== Models ==========
        props = self.llm_manager.get_model_props() or {}
        ordered = self._ordered_models(props)

        st.subheader("📋 利用可能モデル（有効/無効と優先順位）")

        if not props:
            st.warning("モデル情報が取得できませんでした（get_model_props が空）。")
            return

        # 優先順位編集（簡易：順番を上から選び直す方式）
        with st.expander("🧭 優先順位（priority）", expanded=True):
            st.caption("上から順に優先。いったんこの方式で固定し、後でドラッグUIにしたければ差し替え可能。")

            current_priority: List[str] = list(self.state.get("priority") or [])
            # props に存在するものだけで再構成
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
                    # 重複除去しつつ保存
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

        # enabled toggle
        with st.expander("✅ モデルの有効/無効（enabled）", expanded=True):
            enabled_map = self.state.get("enabled_models") or {}
            if not isinstance(enabled_map, dict):
                enabled_map = {}
                self.state["enabled_models"] = enabled_map

            for name in ordered:
                p = props.get(name, {}) or {}
                default_on = (name == "gpt52")  # ★初期思想：gpt52のみ
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

        # quick summary
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
