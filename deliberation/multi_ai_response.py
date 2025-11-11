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

# deliberation/multi_ai_response.py
# ------------------------------------------------------------
# 参加AIモデル一覧（PARTICIPATING_MODELS）
# 本定義は、AI間の審議・比較において、
# 名称と簡単な説明を与えるための静的リストです。
# ------------------------------------------------------------

class MultiAIResponse:
    """
    マルチAI関連の表示と審議をまとめる中核クラス。

    役割:
      - llm_meta から models / judge を整える
      - モデルごとの応答ビューを表示
      - Judge の結果を表示
      - ComposerAI を使って「ベスト候補」を計算し、裏画面で表示

    ※ v1 では Lyra 本体の返答は変更しない。
       あくまでデバッグ＆将来の差し替えのための情報表示に留める。
    """

    def __init__(self) -> None:
        self.display_config = MultiAIDisplayConfig(
            initial={
                "gpt4o": "GPT-4o",
                "hermes": "Hermes",
            }
        )
        self.model_viewer = MultiAIModelViewer(self.display_config)
        self.judge_view = MultiAIJudgeResultView()
        self.judge_ai = JudgeAI()
        self.composer = ComposerAI(mode="winner_only")

    # ---- models / judge の確保 ----
    def _ensure_models(self, llm_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        llm_meta["models"] を返す。
        なければ空 dict を返す（今は Collector までは呼ばない前提）。
        """
        models = llm_meta.get("models")
        if isinstance(models, dict):
            return models
        return {}

    def _ensure_judge(
        self,
        llm_meta: Dict[str, Any],
        models: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        llm_meta["judge"] がなければ JudgeAI を実行し、結果を返す。
        """
        judge = llm_meta.get("judge")
        if isinstance(judge, dict):
            return judge

        if not isinstance(models, dict) or len(models) < 2:
            return None

        judge = self.judge_ai.run(llm_meta)
        return judge

    # ---- メイン描画 ----
    def render(self, llm_meta: Dict[str, Any] | None) -> None:
        if not isinstance(llm_meta, dict) or not llm_meta:
            st.caption("（マルチAI関連のメタ情報がまだありません）")
            return

        # models / judge を確保
        models = self._ensure_models(llm_meta)
        judge = self._ensure_judge(llm_meta, models)

        # 1) 各モデルの応答比較
        with st.expander("🤝 モデル応答比較", expanded=True):
            if models:
                # ここは元々の MultiAIModelViewer の render に合わせる
                self.model_viewer.render(models)
            else:
                st.caption("（models がありません）")

        # 2) Judge の結果
        with st.expander("⚖️ マルチAI審議結果", expanded=True):
            self.judge_view.render(judge)

        # 3) Composer による「ベスト候補」
        with st.expander("🧬 ベスト回答候補（Composer）", expanded=False):
            if not models:
                st.caption("（models がないため、Composer は実行していません）")
                return

            # ベースとなる返答（とりあえず gpt4o 優先）
            base_reply = ""
            if "gpt4o" in models:
                base_reply = str(models["gpt4o"].get("reply") or "")
            else:
                # 先頭モデルをベースにする
                first_key = list(models.keys())[0]
                base_reply = str(models[first_key].get("reply") or "")

            # user_prompt は現段階では必須ではないので空でOK
            final_info = self.composer.decide_final_reply(
                user_prompt="",
                models=models,
                judge=judge,
                base_reply=base_reply,
            )

            # llm_meta にも格納しておく（後で使いたくなった時のため）
            llm_meta["composer"] = final_info

            chosen_model = final_info.get("chosen_model", "unknown")
            final_reply = final_info.get("final_reply") or "（候補なし）"
            mode = final_info.get("mode", "unknown")

            st.markdown(f"- モード: `{mode}`")
            st.markdown(f"- 採用候補モデル: `{chosen_model}`")
            st.markdown("**最終候補テキスト:**")
            st.write(final_reply)
