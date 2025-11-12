# deliberation/multi_ai_response.py

from __future__ import annotations
from typing import Any, Dict, Optional
import streamlit as st

from components.multi_ai_display_config import MultiAIDisplayConfig
from components.multi_ai_model_viewer import MultiAIModelViewer
from components.multi_ai_judge_result_view import MultiAIJudgeResultView
from deliberation.judge_ai import JudgeAI
from deliberation.composer_ai import ComposerAI
from deliberation.participating_models import PARTICIPATING_MODELS


class MultiAIResponse:
    """
    マルチAI関連の表示と審議をまとめる中核クラス。
    """

    def __init__(self) -> None:
        self.display_config = MultiAIDisplayConfig(initial={"gpt4o": "GPT-4o", "hermes": "Hermes"})
        self.model_viewer = MultiAIModelViewer(self.display_config)
        self.judge_view = MultiAIJudgeResultView()
        self.judge_ai = JudgeAI()
        self.composer = ComposerAI(mode="winner_only")

    def _ensure_models(self, llm_meta: Dict[str, Any]) -> Dict[str, Any]:
        models = llm_meta.get("models")
        if isinstance(models, dict):
            return models
        return {}

    def _ensure_judge(self, llm_meta: Dict[str, Any], models: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        judge = llm_meta.get("judge")
        if isinstance(judge, dict):
            return judge
        if not isinstance(models, dict) or len(models) < 2:
            return None
        return self.judge_ai.run(llm_meta)

    def render(self, llm_meta: Dict[str, Any] | None) -> None:
        if not isinstance(llm_meta, dict) or not llm_meta:
            st.caption("（マルチAI関連のメタ情報がまだありません）")
            return

        models = self._ensure_models(llm_meta)
        judge = self._ensure_judge(llm_meta, models)

        with st.expander("🤝 モデル応答比較", expanded=True):
            if models:
                self.model_viewer.render(models)
            else:
                st.caption("（models がありません）")

        with st.expander("⚖️ マルチAI審議結果", expanded=True):
            self.judge_view.render(judge)

        with st.expander("🧬 ベスト回答候補（Composer）", expanded=False):
            if not models:
                st.caption("（models がないため、Composer は実行していません）")
                return

            base_reply = models.get("gpt4o", {}).get("reply") or ""
            final_info = self.composer.decide_final_reply("", models, judge, base_reply)

            llm_meta["composer"] = final_info

            st.markdown(f"- モード: `{final_info.get('mode', 'unknown')}`")
            st.markdown(f"- 採用候補モデル: `{final_info.get('chosen_model', 'unknown')}`")
            st.markdown("**最終候補テキスト:**")
            st.write(final_info.get("final_reply") or "（候補なし）")
