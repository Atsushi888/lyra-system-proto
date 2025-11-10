from typing import Any, Dict, Optional
import streamlit as st


class MultiAIResponse:
    """
    マルチAIの返答を縦に並べて表示するビューアの土台クラス。

    ・まずは MultiModelViewer とほぼ同等の挙動
    ・llm_meta から各モデルの reply を拾って表示する
    ・将来、JudgeAI や Composite の情報をここに足していく前提
    """

    def __init__(
        self,
        title: str = "マルチAIリプライ",
    ) -> None:
        self.title = title

        # ここに追加していくだけでモデルを増やせる
        # key: llm_meta のキー名 / value: 表示ラベル
        self.model_labels: Dict[str, str] = {
            "gpt4o": "GPT-4o",
            "hermes": "Hermes",
            # "claude": "Claude 3" みたいに増やしていく
        }

    def _extract_model_info(
        self,
        llm_meta: Dict[str, Any],
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """
        llm_meta の形が変わっても耐えられるように、
        1か所でモデル情報の取り出し方法をまとめておく。

        想定しているパターン：
        1) llm_meta["gpt4o"] = {"reply": "...", ...}
        2) llm_meta["models"]["gpt4o"] = {"reply": "...", ...}
        """

        if key in llm_meta:
            info = llm_meta.get(key)
        else:
            info = llm_meta.get("models", {}).get(key)

        if not isinstance(info, dict):
            return None
        return info

    def render(self, llm_meta: Dict[str, Any] | None) -> None:
        """
        llm_meta の中身を見て、各モデルの応答を表示する。
        まだレスポンスがない場合は何も出さない。
        """

        # 起動直後など llm_meta が None / 空dict の場合
        if not isinstance(llm_meta, dict) or not llm_meta:
            st.caption("（まだレスポンスがありません）")
            return

        st.markdown(f"### {self.title}")

        # もし prompt_preview があれば、折りたたみで見られるようにしておく
        prompt_preview = llm_meta.get("prompt_preview")
        if isinstance(prompt_preview, str) and prompt_preview.strip():
            with st.expander("📝 プロンプトプレビュー", expanded=False):
                st.code(prompt_preview, language="text")

        has_any = False

        for key, label in self.model_labels.items():
            info = self._extract_model_info(llm_meta, key)
            if not info:
                continue

            has_any = True
            reply = info.get("reply") or info.get("text") or "（返信なし）"

            st.markdown(f"#### {label}")
            st.write(reply)

            # トークン情報などがあれば軽く表示（あればでOK）
            usage = info.get("usage") or info.get("usage_main")
            if isinstance(usage, dict) and usage:
                prompt_tokens = usage.get("prompt_tokens", "？")
                completion_tokens = usage.get("completion_tokens", "？")
                total_tokens = usage.get("total_tokens", "？")
                st.caption(
                    f"tokens: total={total_tokens}, "
                    f"prompt={prompt_tokens}, completion={completion_tokens}"
                )

            st.markdown("---")

        if not has_any:
            st.caption("（表示可能なモデルがありません）")
