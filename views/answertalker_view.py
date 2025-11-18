# views/answertalker_view.py

from __future__ import annotations

from typing import Optional, Dict, Any, List

import streamlit as st

from actors.actor import Actor


class AnswerTalkerView:
    """
    AnswerTalker の動作確認用ビュー（閲覧専用）。

    - Actor インスタンスを受け取り、その中の AnswerTalker に紐づく llm_meta を“見るだけ”
    - 自分から AnswerTalker.run_models() を呼んだりはしない
    - llm_meta["models"] / ["judge"] / ["composer"] を一覧表示する
    """

    def __init__(self, actor: Actor) -> None:
        # いまは直接使っていないが、将来的に
        # Actor 名や Persona 情報などを表示したくなったときのために保持しておく
        self.actor = actor

    def _render_models(self, llm_meta: Dict[str, Any]) -> None:
        st.markdown("### llm_meta に登録された AI 回答一覧（models）")

        models = llm_meta.get("models")
        if not isinstance(models, dict) or not models:
            st.info(
                "llm_meta['models'] に情報がありません。\n\n"
                "- AnswerTalker.run_models() がどこかでまだ実行されていないか、\n"
                "- もしくはモデルから有効な回答が返ってきていない可能性があります。"
            )
            return

        for model_name, info in models.items():
            st.markdown(f"#### モデル: `{model_name}`")

            if isinstance(info, dict):
                status = info.get("status", "unknown")
                text = info.get("text", "")
                err = info.get("error")

                st.write(f"- status: `{status}`")
                if err:
                    st.write(f"- error: `{err}`")

                if text:
                    st.write("**回答テキスト:**")
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

    def _render_judge(self, llm_meta: Dict[str, Any]) -> None:
        st.markdown("### JudgeAI2 の判定結果（llm_meta['judge']）")

        judge = llm_meta.get("judge")
        if not isinstance(judge, dict) or not judge:
            st.info(
                "llm_meta['judge'] に情報がありません。\n\n"
                "- JudgeAI2 がまだ実行されていないか、\n"
                "- あるいはエラーで判定に失敗している可能性があります。"
            )
            return

        status = judge.get("status", "unknown")
        chosen_model = judge.get("chosen_model", "")
        reason = judge.get("reason", "")
        chosen_text = judge.get("chosen_text", "")
        error = judge.get("error")

        st.write(f"- status: `{status}`")
        if chosen_model:
            st.write(f"- chosen_model: `{chosen_model}`")
        if reason:
            st.write(f"- reason: {reason}")
        if error:
            st.write(f"- error: `{error}`")

        if chosen_text:
            st.write("**採用テキスト（chosen_text）:**")
            st.text_area(
                "chosen_text",
                value=chosen_text,
                height=160,
                key="judge_chosen_text",
            )

        # 候補一覧（スコアなど）
        candidates: Any = judge.get("candidates")
        if isinstance(candidates, list) and candidates:
            st.markdown("#### 候補モデル一覧（candidates）")
            for i, c in enumerate(candidates):
                if not isinstance(c, dict):
                    continue
                st.markdown(f"- 候補 {i+1}: `{c.get('model', '')}`")
                st.write(
                    f"  - status: `{c.get('status', 'unknown')}`  "
                    f"/ score: `{c.get('score', '')}`  "
                    f"/ length: `{c.get('length', '')}`"
                )
                if c.get("error"):
                    st.write(f"  - error: `{c.get('error')}`")
                details = c.get("details")
                if isinstance(details, list) and details:
                    st.write("  - details:", ", ".join(str(d) for d in details))
        st.markdown("---")

    def _render_composer(self, llm_meta: Dict[str, Any]) -> None:
        st.markdown("### ComposerAI の最終結果（llm_meta['composer']）")

        comp = llm_meta.get("composer")
        if not isinstance(comp, dict) or not comp:
            st.info(
                "llm_meta['composer'] に情報がありません。\n\n"
                "- ComposerAI がまだ実行されていないか、\n"
                "- あるいはエラーで compose に失敗している可能性があります。"
            )
            return

        status = comp.get("status", "unknown")
        text = comp.get("text", "")
        error = comp.get("error")

        st.write(f"- status: `{status}`")
        if error:
            st.write(f"- error: `{error}`")

        if text:
            st.write("**最終返答テキスト（composer.text）:**")
            st.text_area(
                "composer_text",
                value=text,
                height=180,
                key="composer_text_area",
            )

        st.markdown("---")

    def render(self) -> None:
        st.title("AnswerTalker / ModelsAI・JudgeAI2・Composer デバッグビュー（閲覧専用）")

        st.markdown(
            "この画面では、Actor に紐づく AnswerTalker が保持している "
            "`llm_meta` の内容（models / judge / composer）を参照できます。"
        )
        st.markdown(
            "> ※ この画面からは AnswerTalker.run_models() などは実行しません。"
            " 会談システムや別の処理でパイプラインを走らせたあとに開いてください。"
        )

        st.markdown("---")

        llm_meta: Optional[Dict[str, Any]] = st.session_state.get("llm_meta")
        if not isinstance(llm_meta, dict):
            st.info(
                "llm_meta がまだ初期化されていません。\n\n"
                "- 会談システムや AnswerTalker を利用する処理を一度実行してから、\n"
                "- 再度この画面を開いてください。"
            )
            return

        # Models / Judge / Composer の順に表示
        self._render_models(llm_meta)
        self._render_judge(llm_meta)
        self._render_composer(llm_meta)


# ModeSwitcher などから呼びやすくするラッパ
def render(actor: Actor) -> None:
    view = AnswerTalkerView(actor)
    view.render()
