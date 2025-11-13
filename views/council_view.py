# views/council_view.py

from __future__ import annotations
import streamlit as st

from council.council_manager import CouncilManager


class CouncilView:
    """
    会談システムの薄いラッパー。
    実作業は CouncilManager.render() に丸投げ。
    """
    def __init__( self ):
        self.manager = CouncilManager(st.session_state)

    def render(self) -> None:
        pass
        # self.manager.render()
