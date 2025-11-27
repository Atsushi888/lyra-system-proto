# views/council_view.py

from __future__ import annotations

from typing import Any

import streamlit as st

from council.council_manager import CouncilManager


class CouncilView:
    """
    会談システム画面（β）。
    CouncilManager を session_state に 1 つだけ持って、
    実際の UI 描画は CouncilManager.render() に委譲する薄いラッパ。
    """

    SESSION_MANAGER = "council_manager"

    def __init__(self) -> None:
        # 特に状態は持たないが、将来拡張を見越してクラスとして定義
        pass

    def _get_manager(self) -> CouncilManager:
        """session_state に CouncilManager インスタンスを 1 つだけ確保して返す。"""
        if self.SESSION_MANAGER not in st.session_state:
            st.session_state[self.SESSION_MANAGER] = CouncilManager()
        return st.session_state[self.SESSION_MANAGER]

    def render(self) -> None:
        """ModeSwitcher から呼ばれるエントリポイント。"""
        manager = self._get_manager()
        manager.render()
