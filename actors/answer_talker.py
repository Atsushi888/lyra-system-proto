# actors/answer_talker.py

from __future__ import annotations

from typing import Dict, Any

import streamlit as st

from actors.models import Models


class AnswerTalker:
    """
    回答生成パイプラインのハブ役:
      - models: 各AIから回答を集める
      - judge:  JudgeAI2（将来）
      - composer: 最終整形（将来）

    現段階の責務:
      a. llm_meta の初期設定
      b. Models クラスのコール
    """

    def __init__(self) -> None:
        # Streamlit のセッションから既存 llm_meta を取得
        llm_meta = st.session_state.get("llm_meta")

        if not isinstance(llm_meta, dict):
            # なければ新規初期化
            llm_meta = {
                "models": {},    # 各AIの生回答
                "judge": {},     # JudgeAI2 の結果（現時点では空）
                "composer": {},  # Composer の結果（現時点では空）
            }

        # インスタンス側でも保持
        self.llm_meta: Dict[str, Any] = llm_meta

        # セッションにも戻しておく（他画面からも見えるように）
        st.session_state["llm_meta"] = self.llm_meta

        # Models モジュール初期化
        self.models = Models()

    def run_models(self, user_text: str) -> None:
        """
        Models モジュールを実行し、llm_meta["models"] を更新する。
        いまは「集めるだけ」。Judge/Composer はまだ触らない。
        """
        if not user_text:
            return

        model_results = self.models.collect(user_text)

        if not isinstance(self.llm_meta.get("models"), dict):
            self.llm_meta["models"] = {}

        # 今回は丸ごと置き換えでOK（将来はマージも検討）
        self.llm_meta["models"] = model_results

        # セッションにも反映
        st.session_state["llm_meta"] = self.llm_meta
