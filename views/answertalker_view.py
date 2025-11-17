# views/answertalker_view.py

import streamlit as st
from actors.answer_talker import AnswerTalker


class AnswerTalkerView:
    """AnswerTalker 動作確認用の裏画面。"""

    def __init__(self) -> None:
        # ここで AnswerTalker インスタンスを持つ
        self.talker = AnswerTalker()

    def render(self) -> None:
        st.title("AnswerTalker Backstage")

        st.markdown("#### テスト用入力")
        user_text = st.text_input(
            "プレイヤー発言（テスト用）",
            value="",
            key="answertalker_input",
        )

        if st.button("AnswerTalker を実行", key="answertalker_run"):
            if user_text.strip():
                self.talker.run_models(user_text)
                st.success("AnswerTalker を実行しました。")

        # ここから下は「中身を覗き込みすぎない」表示
        llm_meta = self.talker.llm_meta

        st.markdown("---")
        st.markdown("#### llm_meta の状態（概要のみ）")

        # 初期化されているかどうかだけ表示
        st.write("llm_meta 初期化済み:", isinstance(llm_meta, dict))

        # models の件数だけ軽く見る（中身は表示しない）
        models = llm_meta.get("models") if isinstance(llm_meta, dict) else None
        if isinstance(models, dict):
            st.write("models に登録されている件数:", len(models))
        else:
            st.write("models はまだ dict として初期化されていません。")


def render() -> None:
    """app.py などから呼ぶ用のラッパ。"""
    view = AnswerTalkerView()
    view.render()
