from __future__ import annotations

from typing import Dict, Protocol, Any
import streamlit as st

from auth.roles import Role

from views.user_view import UserView          # いまは未使用でも残しておく
from views.private_view import PrivateView
from views.council_view import CouncilView
from views.llm_manager_view import create_llm_manager_view
from views.answertalker_view import create_answertalker_view
from views.emotion_control_view import create_emotion_control_view
from views.persona_editor_view import create_persona_editor_view
# from views.scene_changer_view import create_scene_changer_view  # ← 封印
from views.narrator_manager_view import create_narrator_manager_view
from views.scene_manager_view import SceneManagerView
from views.dokipower_control_view import create_dokipower_control_view
from views.user_settings_view import create_user_settings_view  # ★ 新規 UserSettings 用


class View(Protocol):
    def render(self) -> None: ...


def _resolve_view(view_or_factory: Any) -> View:
    """
    factory関数 / クラス / インスタンス の違いを吸収して
    安全に View を返すユーティリティ。

    - factory関数        → そのまま呼ぶ
    - クラス（type）     → インスタンス化
    - 生成済みインスタンス → そのまま返す
    """
    try:
        # factory関数
        if callable(view_or_factory) and not isinstance(view_or_factory, type):
            v = view_or_factory()
            return v

        # Viewクラス → new()
        if isinstance(view_or_factory, type):
            return view_or_factory()

        # 生成済みインスタンス
        return view_or_factory

    except Exception as e:
        st.error(f"Viewの生成でエラー: {type(e).__name__}: {e}")
        raise


class ModeSwitcher:
    LABELS: Dict[str, str] = {
        "USER":          "🎛️ ユーザー設定（LLM）",
        "USERSETTINGS":  "💻 ユーザー設定（その他）",
        "PRIVATE":       "⚙️ （※非公開※）",
        "COUNCIL":       "🗣 会談システム（β）",
        "ANSWERTALKER":  "🧩 AnswerTalker（AI統合テスト）",
        "EMOTION":       "💓 感情オーバーライド",
        "PERSONA":       "🖋️ キャラ設定（Persona）",
        # "SCENE":         "🚶‍♀️ シーン移動",  # ← 旧 scene_changer 用ラベルは削除
        "NARRATOR":      "📝 Narrator Debug",
        "SCENEMGR":      "🌏 Scene Emotion Manager",
        "DOKIPOWER":     "💓 ドキドキパワー調整",
    }

    def __init__(self, *, default_key: str = "USER", session_key: str = "view_mode") -> None:
        self.default_key = default_key
        self.session_key = session_key

        self.routes: Dict[str, Dict[str, Any]] = {
            "USER": {
                "label": self.LABELS["USER"],
                "view": create_llm_manager_view,
                "min_role": Role.USER,
            },
            "USERSETTINGS": {   # ★ 新規ルート
                "label": self.LABELS["USERSETTINGS"],
                "view": create_user_settings_view,
                "min_role": Role.USER,
            },
            "PRIVATE": {
                "label": self.LABELS["PRIVATE"],
                "view": PrivateView,
                "min_role": Role.ADMIN,
            },
            "COUNCIL": {
                "label": self.LABELS["COUNCIL"],
                "view": CouncilView,
                "min_role": Role.ADMIN,
            },
            "ANSWERTALKER": {
                "label": self.LABELS["ANSWERTALKER"],
                "view": create_answertalker_view,
                "min_role": Role.ADMIN,
            },
            "EMOTION": {
                "label": self.LABELS["EMOTION"],
                "view": create_emotion_control_view,
                "min_role": Role.ADMIN,
            },
            "PERSONA": {
                "label": self.LABELS["PERSONA"],
                "view": create_persona_editor_view,
                "min_role": Role.ADMIN,
            },
            # "SCENE": {
            #     "label": self.LABELS["SCENE"],
            #     "view": create_scene_changer_view,
            #     "min_role": Role.ADMIN,
            # },
            "NARRATOR": {
                "label": self.LABELS["NARRATOR"],
                "view": create_narrator_manager_view,
                "min_role": Role.ADMIN,
            },
            "SCENEMGR": {
                "label": self.LABELS["SCENEMGR"],
                "view": SceneManagerView,
                "min_role": Role.ADMIN,
            },
            "DOKIPOWER": {
                "label": self.LABELS["DOKIPOWER"],
                "view": create_dokipower_control_view,
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

        view_or_factory: Any = self.routes[cur]["view"]
        view: View = _resolve_view(view_or_factory)
        view.render()
