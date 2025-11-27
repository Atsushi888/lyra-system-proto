# views/council_view.py

from __future__ import annotations

import streamlit as st

from council.council_manager import CouncilManager


SESSION_MANAGER = "council_manager"


def render() -> None:
    """
    会談システム画面エントリポイント。
    CouncilManager のインスタンスを session_state に 1 つだけ持ち、
    画面描画は CouncilManager.render() に委譲する。
    """
    if SESSION_MANAGER not in st.session_state:
        st.session_state[SESSION_MANAGER] = CouncilManager()

    manager: CouncilManager = st.session_state[SESSION_MANAGER]
    manager.render()
