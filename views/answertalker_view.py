# views/answertalker_view.py

from __future__ import annotations

from typing import Optional, Dict, Any

import streamlit as st

from actors.actor import Actor


class AnswerTalkerView:
    """
    AnswerTalker の動作確認用ビュー（閲覧専用）。

    - Actor インスタンスを受け取り、その中の AnswerTalker に紐づく llm_meta を“見るだけ”
    - 自分から AnswerTalker.run_models() を呼んだりはしない
    - llm_meta["models"] に入っている各AIの回答を一覧表示する
    """

    def __init__(self, actor: Actor) -> None:
        self.actor = actor  # いまは未使用だが、将来 Actor 情報を表示したいときのために保持

    def render(self) -> None:
        st.title("AnswerTalker / ModelsAI デバッグビュー（閲覧専用）")

        st.markdown(
            "この画面では、Actor に紐づく AnswerTalker / ModelsAI が収集した "
            "`llm_meta['models']` の内容を参照できます。"
        )
        st.markdown(
            "> ※ この画面からは AnswerTalker.run_models() は実行しません。"
            " 会談システムや別の処理でパイプラインを走らせたあとに開いてください。"
        )

        st.markdown("---")
        st.markdown("### llm_meta に登録された AI 回答一覧（models）")

        llm_meta: Optional[Dict[str, Any]] = st.session_state.get("llm_meta")
        if not isinstance(llm_meta, dict):
            st.info(
                "llm_meta がまだ初期化されていません。\n\n"
                "- 会談システムや AnswerTalker を利用する処理を一度実行してから、\n"
                "- 再度この画面を開いてください。"
            )
            return

        models = llm_meta.get("models")
        if not isinstance(models, dict) or not models:
            st.info(
                "llm_meta['models'] に情報がありません。\n\n"
                "- AnswerTalker.run_models() がどこかでまだ実行されていないか、\n"
                "- もしくはモデルから有効な回答が返ってきていない可能性があります。"
            )
            return

        # ★ 各 AI の回答をシンプルに表示（json.dumps などは使わない）
        for model_name, info in models.items():
            st.markdown(f"#### モデル: `{model_name}`")

            if isinstance(info, dict):
                status = info.get("status", "unknown")
                text = info.get("text", "")
                st.write(f"- status: `{status}`")

                if text:
                    st.write("**回答テキスト:**")
                    # 長すぎるときは少しだけ切って表示
                    if len(text) > 400:
                        st.text_area(
                            "text（先頭400文字）",
                            value=text[:400] + " ...",
                            height=120,
                            key=f"models_text_{model_name}",
                        )
                    else:
                        st.text_area(
                            "text",
                            value=text,
                            height=120,
                            key=f"models_text_{model_name}",
                        )
                else:
                    st.write("（text フィールドがありません）")
            else:
                st.write("想定外の形式です:", info)

            st.markdown("---")
            if isinstance(info, dict):
                status = info.get("status", "unknown")
                text = info.get("text", "")
                err  = info.get("error")
            
                st.write(f"- status: `{status}`")
                if err:
                    st.write(f"- error: `{err}`")
            st.markdown("---")


# ModeSwitcher などから呼びやすくするラッパ
def render(actor: Actor) -> None:
    view = AnswerTalkerView(actor)
    view.render()
