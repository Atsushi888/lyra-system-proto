# components/mode_switcher.py
from __future__ import annotations
from typing import Dict, Protocol, Optional
import streamlit as st

# 露出ポリシーは内部で判定
from utils.access import is_admin
from utils.feature_flags import flag, SHOW_USER_WINDOW_DEFAULT, SHOW_BACKSTAGE_DEFAULT

# View 実体はここで生成
from views.game_view import GameView
from views.user_view import UserView
from views.backstage_view import BackstageView
from views.private_view import PrivateView


class View(Protocol):
    def render(self) -> None: ...


class ModeSwitcher:
    """
    ページ初期設定／モード露出／LABELS／View生成／左サイド表示／遷移／中央描画まで
    “ボタン関連の業務”を完全担当するワンストップ・クラス。
    """
    # ラベルもここに集約
    LABELS: Dict[str, str] = {
        "PLAY":      "🎮 ゲームモード",
        "USER":      "🎛️ ユーザー設定",
        "BACKSTAGE": "🧠 AIリプライシステム",
        "PRIVATE":   "⚙️ （※非公開※）",
    }

    def __init__(
        self,
        *,
        page_title: str = "Lyra System",
        session_key: str = "view_mode",
        default_mode: str = "PLAY",
        sidebar_title: str = "画面切替",
    ) -> None:
        # 1) ページ初期設定
        st.set_page_config(page_title=page_title, layout="wide")

        self.session_key   = session_key
        self.default_mode  = default_mode
        self.sidebar_title = sidebar_title

        # 2) 権限・フラグから露出可否を判定（ここで完結）
        admin          = is_admin()
        show_private   = admin and flag("SHOW_USER_WINDOW", SHOW_USER_WINDOW_DEFAULT)
        show_backstage = admin and flag("SHOW_BACKSTAGE",  SHOW_BACKSTAGE_DEFAULT)

        # 3) View をここで生成＆露出フィルタ
        views_all: Dict[str, View] = {
            "PLAY":      GameView(),
            "USER":      UserView(),
            "BACKSTAGE": BackstageView(),
            "PRIVATE":   PrivateView(),
        }
        allowed = ["PLAY", "USER"]
        if show_backstage: allowed.append("BACKSTAGE")
        if show_private:   allowed.append("PRIVATE")

        self.routes: Dict[str, View] = {k: views_all[k] for k in allowed}

        # default が非公開で消えた場合に備えて保険
        if self.default_mode not in self.routes:
            self.default_mode = next(iter(self.routes.keys()))

        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = self.default_mode

    # 現在モード取得（権限変化で外れたらdefaultへ戻す）
    @property
    def current(self) -> str:
        cur = st.session_state.get(self.session_key, self.default_mode)
        if cur not in self.routes:
            cur = self.default_mode
            st.session_state[self.session_key] = cur
        return cur

    # 左サイド：ボタンの表示・遷移・現在表示（全部ここ）
    def _render_sidebar(self) -> None:
        st.sidebar.markdown(f"## {self.sidebar_title}")
        cur = self.current
        for key, view in self.routes.items():
            label = self.LABELS.get(key, key)
            disabled = (key == cur)
            if st.sidebar.button(label, use_container_width=True, disabled=disabled, key=f"mode_{key}"):
                st.session_state[self.session_key] = key
                st.rerun()
        st.sidebar.caption(f"現在: {self.LABELS.get(cur, cur)}")

    # 中央：見出し＋View描画（表示責務もここ）
    def _render_center(self) -> None:
        key = self.current
        st.subheader(self.LABELS.get(key, key))
        self.routes[key].render()

    # ワンストップ呼び出し
    def render(self) -> None:
        self._render_sidebar()
        self._render_center()
