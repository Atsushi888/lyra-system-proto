# mode_switcher.py

from __future__ import annotations
from typing import Dict, Protocol, Any

import streamlit as st

from auth.roles import Role

from views.game_view import GameView
from views.backstage_view import BackstageView
from views.private_view import PrivateView
from views.council_view import CouncilView
from council.council_manager import create_answertalker_view   # 既存
from views.llm_manager_view import create_llm_manager_view     # ★ 追加：LLM 用ファクトリ


class View(Protocol):
    def render(self) -> None: ...


class ModeSwitcher:
    """
    表示切替のみ担当（認証ロジックは持たない）。
    routes は __init__ 内で内蔵生成。
    """

    LABELS: Dict[str, str] = {
        "PLAY":          "🎮 ゲームモード",
        "USER":          "🎛️ ユーザー設定（LLM）",
        "BACKSTAGE":     "🧠 AIリプライシステム",
        "PRIVATE":       "⚙️ （※非公開※）",
        "COUNCIL":       "🗣 会談システム（β）",
        "ANSWERTALKER":  "🧩 AnswerTalker（AI統合テスト）",
    }

    def __init__(self, *, default_key: str = "PLAY", session_key: str = "view_mode") -> None:
        self.default_key = default_key
        self.session_key = session_key

        # 内蔵ルーティング
        # view には「インスタンス」か「ビューを返す関数(callable)」のどちらかを入れてよい。
        self.routes: Dict[str, Dict[str, Any]] = {
            "PLAY": {
                "label": self.LABELS["PLAY"],
                "view": GameView(),              # インスタンス
                "min_role": Role.USER,
            },
            "USER": {
                "label": self.LABELS["USER"],
                # ★ UserView は廃止し、LLMManager 用ファクトリに差し替え
                "view": create_llm_manager_view,
                "min_role": Role.USER,
            },
            "BACKSTAGE": {
                "label": self.LABELS["BACKSTAGE"],
                "view": BackstageView(),
                "min_role": Role.ADMIN,
            },
            "PRIVATE": {
                "label": self.LABELS["PRIVATE"],
                "view": PrivateView(),
                "min_role": Role.ADMIN,
            },
            "COUNCIL": {
                "label": self.LABELS["COUNCIL"],
                "view": CouncilView(),
                "min_role": Role.ADMIN,
            },
            "ANSWERTALKER": {
                "label": self.LABELS["ANSWERTALKER"],
                "view": create_answertalker_view,   # ファクトリ関数
                "min_role": Role.ADMIN,
            },
        }

        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = self.default_key

    @property
    def current(self) -> str:
        cur = st.session_state.get(self.session_key, self.default_key)
        if cur not in self.routes:
            cur = self.default_key
            st.session_state[self.session_key] = cur
        return cur

    def render(self, user_role: Role) -> None:
        st.sidebar.markdown("## 画面切替")

        visible_keys = [
            k for k, cfg in self.routes.items()
            if user_role >= cfg.get("min_role", Role.USER)
        ]

        cur = self.current
        if cur not in visible_keys and visible_keys:
            cur = visible_keys[0]
            st.session_state[self.session_key] = cur

        for key in visible_keys:
            label = self.routes[key]["label"]
            disabled = (key == cur)
            if st.sidebar.button(
                label,
                use_container_width=True,
                disabled=disabled,
                key=f"mode_{key}",
            ):
                st.session_state[self.session_key] = key
                st.rerun()

        if visible_keys:
            st.sidebar.caption(f"現在: {self.routes[cur]['label']}")
        else:
            st.sidebar.warning("アクセス可能な画面がありません。")

        if not visible_keys:
            return

        st.subheader(self.routes[cur]["label"])

        view_or_factory = self.routes[cur]["view"]

        # インスタンス or ファクトリ関数の両対応
        if callable(view_or_factory):
            view: View = view_or_factory()
        else:
            view: View = view_or_factory

        view.render()
