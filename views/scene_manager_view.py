# views/scene_manager_view.py
from __future__ import annotations

import streamlit as st

from actors.scene.scene_manager import SceneManager


class SceneManagerView:
    """
    SceneManager の UI ラッパ。
    - Streamlit の session_state に SceneManager を 1個だけ保持
    - 生成時に JSON を load()
    - render() で SceneManager.render() をそのまま呼ぶ
    """

    SESSION_KEY = "scene_manager_instance"

    def __init__(self) -> None:
        # まだ SceneManager がなければ作って JSON を読み込む
        if self.SESSION_KEY not in st.session_state:
            mgr = SceneManager(
                path="actors/scene/scene_bonus/scene_emotion_map.json"
            )
            mgr.load()  # ← ここで JSON を読む
            st.session_state[self.SESSION_KEY] = mgr

        # 以降は同じインスタンスを使い続ける
        self.manager: SceneManager = st.session_state[self.SESSION_KEY]

    def render(self) -> None:
        # 実際の UI は SceneManager 側に任せる
        self.manager.render()
