# actors/models_ai.py
from __future__ import annotations

from typing import Dict, Any, List

from llm.llm_manager import LLMManager


class ModelsAI:
    """
    LLMManager を経由して、登録済みモデル（gpt4o / gpt51 / hermes など）
    からまとめて回答を集めるヘルパ。

    - AnswerTalker から LLMManager インスタンスを受け取る
    - LLMManager.get_model_props() でモデル一覧を取得
    - 各モデルについて LLMManager.call_model(...) を実行
    """

    def __init__(self, llm_manager: LLMManager) -> None:
        self.llm_manager = llm_manager
        # AnswerTalker から渡された LLMManager からモデル定義を取ってくる
        self.model_props: Dict[str, Dict[str, Any]] = (
            llm_manager.get_model_props() or {}
        )

    # ============================
    # 内部: 結果の正規化
    # ============================
    def _normalize_result(self, result: Any) -> Dict[str, Any]:
        """
        LLMRouter / LLMManager から返ってきた結果を、
        AnswerTalkerView が扱いやすい dict 形式にそろえる。

        - (text, usage) タプル or text だけ、どちらにも対応
        - text が空 or 空白のみの場合は "empty_response" エラー扱いにする
        """
        reply_text = None
        usage = None
        meta: Dict[str, Any] = {}

        # もともとの形式を吸収
        if isinstance(result, tuple):
            if len(result) >= 2:
                reply_text, usage = result[:2]
            elif len(result) == 1:
                reply_text = result[0]
        else:
            reply_text = result

        text = "" if reply_text is None else str(reply_text)
        text_stripped = text.strip()

        # ★ GPT-5.1 の「content が空」のケースをここで検出して error 扱いにする
        if not text_stripped:
            return {
                "text": "",
                "usage": usage,
                "meta": meta,
                "status": "error",
                "error": "empty_response",
            }

        return {
            "text": text,
            "usage": usage,
            "meta": meta,
            "status": "ok",
        }

    # ============================
    # 公開: 複数モデルから回答収集
    # ============================
    def collect(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Persona.build_messages() で組み立てた messages を全モデルに投げ、
        { model_name: {status, text, usage, meta, error?} } という dict を返す。
        """
        results: Dict[str, Any] = {}

        # LLMManager 側の model_props を毎回取り直してもよいが、
        # いまは初期化時に取得したものを使う。
        model_props = self.model_props

        for name, props in model_props.items():
            enabled = props.get("enabled", True)
            if not enabled:
                results[name] = {
                    "status": "disabled",
                    "text": "",
                    "usage": None,
                    "meta": {},
                    "error": "disabled_by_config",
                }
                continue

            try:
                # LLMManager に丸投げ
                raw = self.llm_manager.call_model(
                    model_name=name,
                    messages=messages,
                )
                norm = self._normalize_result(raw)

                # _normalize_result が status を持っていなければ ok 扱い
                if "status" not in norm:
                    norm["status"] = "ok"

                results[name] = norm
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "text": "",
                    "usage": None,
                    "meta": {},
                    "error": str(e),
                }

        return results
