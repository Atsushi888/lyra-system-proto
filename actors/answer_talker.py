# actors/answer_talker.py

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from actors.models_ai import ModelsAI
from actors.judge_ai2 import JudgeAI2
from actors.composer_ai import ComposerAI


class AnswerTalker:
    """
    AI回答パイプラインの司令塔クラス。

    - ModelsAI:   複数モデルから回答収集（gpt4o / gpt51 / hermes など）
    - JudgeAI2:   どのモデルの回答を採用するかを決定
    - ComposerAI: 採用候補をもとに最終的な返答テキストを生成

    ※ このクラスは「一次返答（reply_text）」を受け取らない。
       Persona と conversation_log から組み立てた messages を入力として、
       自前で Models → Judge → Composer を回し、最終的な返答テキストを返す。
    """

    def __init__(self) -> None:
        # ここを「モデル定義の唯一のソース」にする
        # enabled        : このモデルを使うかどうか
        # priority       : JudgeAI2 での優先度（大きいほど優先）
        # router_fn      : LLMRouter 上のメソッド名
        # label          : 表示用ラベル（デバッグ用）
        self.model_props: Dict[str, Dict[str, Any]] = {
            "gpt4o": {
                "enabled": True,
                "priority": 3,
                "router_fn": "call_gpt4o",
                "label": "GPT-4o",
            },
            "gpt51": {
                "enabled": True,
                "priority": 2,
                "router_fn": "call_gpt51",
                "label": "GPT-5.1",
            },
            "hermes": {
                "enabled": True,
                "priority": 1,
                "router_fn": "call_hermes",
                "label": "Hermes 4",
            },
        }

        # llm_meta の初期化（session_state 経由で共有）
        llm_meta = st.session_state.get("llm_meta")
        if not isinstance(llm_meta, dict):
            llm_meta = {
                "models": {},    # ModelsAI.collect の結果
                "judge": {},     # JudgeAI2.process の結果
                "composer": {},  # ComposerAI.compose の結果
            }

        self.llm_meta: Dict[str, Any] = llm_meta
        st.session_state["llm_meta"] = self.llm_meta

        # サブモジュールに model_props を渡す
        self.models_ai = ModelsAI(self.model_props)
        self.judge_ai = JudgeAI2(self.model_props)
        self.composer_ai = ComposerAI()

    # ============================
    # ModelsAI 呼び出し
    # ============================
    def run_models(self, messages: List[Dict[str, str]]) -> None:
        """
        Persona.build_messages() で組み立てた messages を受け取り、
        ModelsAI.collect() を実行して llm_meta["models"] を更新する。
        """
        if not messages:
            return

        results = self.models_ai.collect(messages)
        self.llm_meta["models"] = results
        st.session_state["llm_meta"] = self.llm_meta

    # ============================
    # 公開インターフェース
    # ============================
    def speak(
        self,
        messages: List[Dict[str, str]],
        user_text: str = "",
    ) -> str:
        """
        Actor から messages（および任意で user_text）を受け取り、
        - ModelsAI.collect
        - JudgeAI2.process
        - ComposerAI.compose
        を順に実行して「最終返答テキスト」を返す。

        戻り値:
            final_text: str
        """
        if not messages:
            # messages が空なら何もできないので空文字を返す
            return ""

        # ① 複数モデルの回答収集
        self.run_models(messages)

        # ② JudgeAI2 による採択
        try:
            judge_result = self.judge_ai.process(self.llm_meta)
        except Exception as e:
            judge_result = {
                "status": "error",
                "error": str(e),
                "chosen_model": "",
                "chosen_text": "",
            }
        self.llm_meta["judge"] = judge_result

        # ③ ComposerAI による仕上げ
        try:
            # ComposerAI の実装次第で user_text / messages を渡したくなったら
            # シグネチャを拡張する想定（現状は llm_meta のみ）
            composed = self.composer_ai.compose(self.llm_meta)
        except Exception as e:
            # 失敗時は Judge の chosen_text をフォールバックとして使う
            fallback = ""
            if isinstance(judge_result, dict):
                fallback = judge_result.get("chosen_text") or ""
            composed = {
                "status": "error",
                "error": str(e),
                "text": fallback,
            }

        self.llm_meta["composer"] = composed
        st.session_state["llm_meta"] = self.llm_meta

        # ④ 最終返答テキストの決定
        final_text = ""
        if isinstance(composed, dict):
            final_text = composed.get("text") or ""
        if not final_text and isinstance(judge_result, dict):
            final_text = judge_result.get("chosen_text") or ""

        return final_text
