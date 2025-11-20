# views/llm_manager_view.py

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from llm.llm_manager import LLMManager


def _bool_emoji(v: bool) -> str:
    return "✅" if v else "❌"


class LLMManagerView:
    """
    LLMManager に関する情報を表示する専用ビュー。

    主な表示内容:
      - persona_id と紐づく LLMManager の状態サマリ
      - 登録済みモデル一覧（enabled / available / missing env など）
      - llm_default.yaml の編集ガイド
    """

    def __init__(self, manager: LLMManager) -> None:
        self.manager = manager

    # -----------------------------
    # 内部: モデル状態表示
    # -----------------------------
    def _render_llm_status(self) -> None:
        st.markdown("### LLM 設定・接続状態")

        summary: Dict[str, Any] = self.manager.get_status_summary()
        persona_id = summary.get("persona_id", "(unknown)")
        st.write(f"- persona_id: `{persona_id}`")

        models: List[Dict[str, Any]] = summary.get("models") or []
        if not models:
            st.info("現在、LLMManager に登録されているモデルがありません。")
            st.markdown("---")
            return

        st.markdown("#### 利用可能な LLM 一覧")

        for m in models:
            name = m.get("name", "")
            label = m.get("label", name)
            vendor = m.get("vendor", "unknown")
            enabled = bool(m.get("enabled", False))
            available = bool(m.get("available", False))
            roles = m.get("roles") or []
            missing_env = m.get("missing_env") or []
            priority = m.get("priority", None)

            st.markdown(f"##### モデル: `{label}` (`{name}`)")

            cols = st.columns(3)
            with cols[0]:
                st.write(f"- vendor: `{vendor}`")
                if priority is not None:
                    st.write(f"- priority: `{priority}`")
            with cols[1]:
                st.write(f"- enabled: {_bool_emoji(enabled)}")
            with cols[2]:
                st.write(f"- available (env OK): {_bool_emoji(available)}")

            st.write(f"- roles: `{', '.join(roles) if roles else '(none)'}`")

            if missing_env:
                st.warning(
                    "不足している環境変数:\n"
                    + "\n".join(f"- `{k}`" for k in missing_env)
                )
            else:
                st.caption("必要な環境変数はすべて設定済みです。")

            st.markdown("---")

    # -----------------------------
    # 内部: llm_default.yaml ガイド
    # -----------------------------
    def _render_llm_config_guide(self) -> None:
        st.markdown("### LLM 設定ファイル（llm_default.yaml）について")

        st.markdown(
            "Lyra-System の LLM 構成は、原則として "
            "`config/llm_default.yaml` で定義します。\n\n"
            "このファイルを編集することで、どの LLM を使うか、"
            "優先度や役割（main / refiner / memory など）を簡単に切り替えられます。"
        )

        with st.expander("llm_default.yaml のサンプル（参考）"):
            sample_yaml = """\
models:
  gpt4o:
    label: "GPT-4o"
    router_fn: "call_gpt4o"
    priority: 3.0
    vendor: "openai"
    roles: ["main", "refiner"]
    required_env: ["OPENAI_API_KEY"]
    enabled: true
    default_params:
      temperature: 0.9
      max_tokens: 900

  gpt51:
    label: "GPT-5.1"
    router_fn: "call_gpt51"
    priority: 4.0
    vendor: "openai"
    roles: ["main", "memory"]
    required_env: ["OPENAI_API_KEY", "GPT51_MODEL"]
    enabled: true
    default_params:
      temperature: 0.95
      max_tokens: 900

  hermes:
    label: "Hermes 4"
    router_fn: "call_hermes"
    priority: 2.0
    vendor: "openrouter"
    roles: ["main"]
    required_env: ["OPENROUTER_API_KEY"]
    enabled: true
    default_params:
      temperature: 0.8
      max_tokens: 900
"""
            st.code(sample_yaml, language="yaml")

        st.markdown(
            "- `router_fn`: LLMRouter 内のメソッド名（例: `call_gpt4o`）\n"
            "- `roles`: このモデルをどの用途で使うか（main/refiner/memory など）\n"
            "- `required_env`: 必要な環境変数。未設定だと `available=False` になります。"
        )

        st.markdown("---")

    # -----------------------------
    # 公開 render
    # -----------------------------
    def render(self) -> None:
        """
        LLMManager 関連の情報をまとめて表示する。
        """
        self._render_llm_status()
        self._render_llm_config_guide()
