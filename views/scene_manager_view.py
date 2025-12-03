from __future__ import annotations

import streamlit as st

from actors.scene.scene_manager import SceneManager


class SceneManagerView:
    """
    SceneManager の UI ラッパ。
    """

    SESSION_KEY = "scene_manager_instance"

    def __init__(self) -> None:
        if self.SESSION_KEY not in st.session_state:
            mgr = SceneManager(
                path="actors/scene/scene_bonus/scene_emotion_map.json"
            )
            mgr.load()
            st.session_state[self.SESSION_KEY] = mgr

        self.manager: SceneManager = st.session_state[self.SESSION_KEY]

    def render(self) -> None:
        self.manager.render()
