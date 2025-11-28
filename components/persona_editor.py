# components/persona_editor.py
from __future__ import annotations

from typing import Optional

import streamlit as st

from personas.persona_floria_ja import Persona


class PersonaEditor:
    """
    フローリアなどの Persona 情報を編集するための UI コンポーネント（プロトタイプ）。

    現段階では：
      - Persona の現在値を表示するだけ
      - 保存や JSON 出力はまだ実装しない（骨組みのみ）
    """

    def __init__(
        self,
        *,
        persona: Optional[Persona] = None,
        session_key: str = "persona_editor_state",
    ) -> None:
        self.session_key = session_key

        # いまはフローリア固定で十分。将来的に選択式に拡張可能。
        self.persona: Persona = persona or Persona()

    def render(self) -> None:
        st.markdown("## 🖋️ キャラ設定編集（Persona Prototype）")
        st.caption(
            "※ 現在は Persona 情報の閲覧のみ対応しています。\n"
            "　JSON への保存や、Persona への書き戻し機能は今後追加予定です。"
        )

        # ---- 基本情報 ----
        with st.expander("基本情報（読み取り専用）", expanded=True):
            st.text_input("キャラクターID", value=self.persona.char_id, disabled=True)
            st.text_input("名前", value=self.persona.name, disabled=True)

        # ---- system_prompt ----
        with st.expander("system_prompt（ロール指示・読み取り専用）", expanded=False):
            st.text_area(
                "system_prompt",
                value=self.persona.system_prompt,
                height=200,
                disabled=True,
            )

        # ---- starter_hint ----
        with st.expander("starter_hint（会話開始時ヒント・読み取り専用）", expanded=False):
            st.text_area(
                "starter_hint",
                value=self.persona.starter_hint,
                height=120,
                disabled=True,
            )

        # ---- style_hint ----
        with st.expander("style_hint（文体メモ・読み取り専用）", expanded=True):
            st.text_area(
                "style_hint",
                value=self.persona.style_hint,
                height=220,
                disabled=True,
            )

        st.markdown("---")
        st.info(
            "この Persona 編集画面は、まだ『閲覧専用モード』です。\n"
            "次のステップとして：\n"
            " - style_hint / system_prompt の編集\n"
            " - JSON への保存と読み込み\n"
            " - AnswerTalker / Refiner への即時反映\n"
            "などを順次追加していく予定です。"
        )
