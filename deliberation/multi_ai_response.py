# components/multi_ai_response.py

from __future__ import annotations

from typing import Any, Dict, Optional, List
import streamlit as st

from components.multi_ai_display_config import MultiAIDisplayConfig
from components.multi_ai_model_viewer import MultiAIModelViewer
from components.multi_ai_judge_result_view import MultiAIJudgeResultView
from deliberation.judge_ai import JudgeAI  # パスは環境に合わせて

PARTICIPATING_MODELS = {
    "gpt4o": "GPT-4o",
    "hermes": "Hermes",
}

class MultiAIResponse:
    """
    マルチAIレスポンスシステムの中核クラス。

    ・表示対象AIの設定（MultiAIDisplayConfig）
    ・モデル応答ビュー（MultiAIModelViewer）
    ・JudgeAI による審議実行
    ・審議結果ビュー（MultiAIJudgeResultView）

    DebugPanel などの上位側は、このクラスに llm_meta を渡して
    render() を呼ぶだけでよい。

    ※ llm_meta["models"] が無い場合、最後の assistant 発言から
       GPT-4o の仮 models を組み立てるフォールバックも持つ。
    """

    def __init__(self) -> None:
        display_config = MultiAIDisplayConfig( initial=PARTICIPATING_MODELS )
        self.model_viewer = MultiAIModelViewer(display_config)
        self.judge_view = MultiAIJudgeResultView()
        self.judge_ai = JudgeAI()

    # ===== フォールバック: models が無いとき自力で組み立てる =====
    def _fallback_models_from_state(
        self,
        llm_meta: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        llm_meta["models"] が存在しないとき、
        st.session_state["messages"] から最後の assistant 発言を拾って
        GPT-4o の仮 models を作る。

        これは「とりあえず裏画面で中身を見たい」ための保険。
        本命は conversation_engine.py 側で models を詰めること。
        """
        try:
            messages: List[Dict[str, str]] = st.session_state.get("messages", [])
            last_assistant = None
            for m in reversed(messages):
                if m.get("role") == "assistant":
                    last_assistant = m.get("content", "")
                    break

            if not last_assistant:
                return None

            usage_main = llm_meta.get("usage_main") or llm_meta.get("usage") or {}

            models = {
                "gpt4o": {
                    "reply": last_assistant,
                    "usage": usage_main,
                    "route": llm_meta.get("route", "gpt"),
                    "model_name": llm_meta.get("model_main", "gpt-4o"),
                }
            }
            return models
        except Exception:
            return None

# components/multi_ai_response.py の中

class MultiAIResponse:
    ...

    def _ensure_models(self, llm_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        models = llm_meta.get("models")
        if isinstance(models, dict) and models:
            return models
        return None
            
    def _ensure_judge(self, llm_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        llm_meta の状態を見て、必要であれば JudgeAI を実行し、
        llm_meta["judge"] を埋めて返す。
        """
        if not isinstance(llm_meta, dict):
            return {"winner": None, "reason": "llm_meta not available"}
    
        # 既に judge が dict で存在すればそのまま使う
        judge = llm_meta.get("judge")
        if isinstance(judge, dict):
            return judge
    
        # models が無い or 少なければ判定できない
        models = llm_meta.get("models")
        if not isinstance(models, dict) or len(models) < 2:
            return {
                "winner": None,
                "reason": "有効なモデル数が不足しています。",
                "score_diff": 0.0,
            }
    
        # ここで初めて判定実行
        try:
            judge = self.judge_ai.run(llm_meta)
            return judge if isinstance(judge, dict) else {
                "winner": None,
                "reason": "JudgeAI returned invalid data.",
                "score_diff": 0.0,
            }
        except Exception as e:
            return {
                "winner": None,
                "reason": f"JudgeAI 実行中にエラー: {e}",
                "score_diff": 0.0,
            }

    def render(self, llm_meta: Optional[Dict[str, Any]]) -> None:
        if not isinstance(llm_meta, dict) or not llm_meta:
            st.caption("（まだマルチAIレスポンスはありません）")
            return

        st.markdown("### ✒️ マルチAIレスポンス")

        # プロンプトプレビュー
        ...

        # モデル応答比較
        models = self._ensure_models(llm_meta)
        if models:
            with st.expander("🤝 モデル応答比較", expanded=True):
                self.model_viewer.render(models)
        else:
            st.caption("（models 情報がありません）")

        # Judge
        judge = self._ensure_judge(llm_meta)
        with st.expander("⚖️ マルチAI審議結果", expanded=True):
            self.judge_view.render(judge)
