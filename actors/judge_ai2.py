# actors/judge_ai2.py

from __future__ import annotations

from typing import Dict, Any, List, Optional


class JudgeAI2:
    """
    ModelsAI が集めた複数モデルの回答を評価して、
    - 採用モデル
    - 採用テキスト
    - 数値ベースの理由（reason）
    - 文章コメント付きの理由（reason_text）
    - 各候補モデルの詳細（candidates）
    を llm_meta["judge"] 形式で返すクラス。

    llm_meta['judge'] の想定フォーマット:
    {
        "status": "ok" | "error",
        "chosen_model": "gpt51",
        "reason": "status_ok / length_bonus_8.6 / priority_bonus_4.0",
        "reason_text": "GPT-5.1 のスコアが最も高く ... ため採用しました。",
        "chosen_text": "<実際に採用されたテキスト>",
        "candidates": [
            {
                "model": "gpt4o",
                "status": "ok",
                "score": 18.72,
                "length": 272,
                "details": [
                    "status_ok",
                    "length_bonus_2.7",
                    "priority_bonus_6.0",
                ],
                "error": None,
            },
            ...
        ],
        "error": None or "<エラーメッセージ>",
    }
    """

    def __init__(self, model_priority: Optional[Dict[str, float]] = None) -> None:
        # モデルごとの「優先度ボーナス」
        # （値が大きいほどスコアにプラスされる）
        self.model_priority: Dict[str, float] = model_priority or {
            "gpt4o": 4.0,
            "gpt51": 6.0,
            "hermes": 2.0,
        }

    # ===== 公開 API（メイン） ===================================

    def judge(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """
        models: llm_meta["models"] をそのまま受け取る。

        戻り値: llm_meta["judge"] にそのまま入れられる dict。
        """
        if not isinstance(models, dict) or not models:
            return {
                "status": "error",
                "chosen_model": None,
                "reason": "no_models",
                "reason_text": "評価対象となるモデル回答が存在しなかったため、判定を行えませんでした。",
                "chosen_text": "",
                "candidates": [],
                "error": "models is empty",
            }

        candidates: List[Dict[str, Any]] = []

        for model_name, info in models.items():
            if not isinstance(info, dict):
                candidates.append(
                    {
                        "model": model_name,
                        "status": "error",
                        "score": 0.0,
                        "length": 0,
                        "details": ["invalid_info"],
                        "error": "model info is not dict",
                    }
                )
                continue

            status = info.get("status", "unknown")
            text = info.get("text") or ""
            error = info.get("error")

            length = len(text)
            base_details: List[str] = []
            score = 0.0

            if status == "ok" and text.strip():
                base_details.append("status_ok")

                # テキスト長に応じたボーナス：ざっくり 100 文字≒1pt, 上限 10pt
                length_bonus = min(length / 100.0, 10.0)
                score += length_bonus
                base_details.append(f"length_bonus_{length_bonus:.1f}")

                # モデルごとの優先度ボーナス
                priority_bonus = float(self.model_priority.get(model_name, 1.0))
                score += priority_bonus
                base_details.append(f"priority_bonus_{priority_bonus:.1f}")
            else:
                # エラーや空文字のとき
                base_details.append("status_ng")
                if status != "ok":
                    base_details.append(f"status_{status}")
                if error:
                    base_details.append("has_error")
                # score は 0 のまま

            candidates.append(
                {
                    "model": model_name,
                    "status": status,
                    "score": round(score, 2),
                    "length": length,
                    "details": base_details,
                    "error": error,
                }
            )

        # 有効な候補がなければエラー扱い
        if not candidates:
            return {
                "status": "error",
                "chosen_model": None,
                "reason": "no_candidates",
                "reason_text": "有効な候補モデルが 1 つもなかったため、判定を行えませんでした。",
                "chosen_text": "",
                "candidates": [],
                "error": "no candidates",
            }

        # スコアで最大のものを選択（同点の場合は長さ→名前順でタイブレーク）
        chosen = max(
            candidates,
            key=lambda c: (
                c.get("score", 0.0),
                c.get("length", 0),
                str(c.get("model", "")),
            ),
        )
        chosen_model = chosen.get("model")
        chosen_info = models.get(chosen_model, {}) if chosen_model else {}
        chosen_text = (chosen_info.get("text") or "") if isinstance(chosen_info, dict) else ""

        # 数値ベースの reason（従来の文字列）
        reason_numeric = " / ".join(chosen.get("details") or [])

        # 文章コメント付きの reason_text
        reason_sentence = self._build_reason_sentence(chosen, candidates)

        return {
            "status": "ok",
            "chosen_model": chosen_model,
            "reason": reason_numeric,
            "reason_text": reason_sentence,
            "chosen_text": chosen_text,
            "candidates": candidates,
            "error": None,
        }

    # ===== 後方互換用ラッパ（既存コード対策） =====================

    def run(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """run(...) で呼ばれても judge(...) に委譲。"""
        return self.judge(models)

    def process(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """process(...) で呼ばれても judge(...) に委譲。"""
        return self.judge(models)

    def process_models(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """process_models(...) で呼ばれても judge(...) に委譲。"""
        return self.judge(models)

    def process_single_result(self, result: Any) -> Dict[str, Any]:
        """
        旧 JudgeAI 互換。
        - result が {"models": {...}} 形式なら、その models を使って判定。
        - それ以外なら「単一回答だけを候補にした擬似判定」を行う。
        """
        if isinstance(result, dict) and isinstance(result.get("models"), dict):
            return self.judge(result["models"])

        # 単一回答用のダミー models
        text = ""
        if isinstance(result, (list, tuple)) and result:
            text = str(result[0])
        elif isinstance(result, str):
            text = result

        models = {
            "gpt4o": {
                "status": "ok" if text.strip() else "error",
                "text": text,
                "error": None if text.strip() else "empty text",
            }
        }
        return self.judge(models)

    # ===== 内部ヘルパ ============================================

    def _build_reason_sentence(
        self,
        chosen: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> str:
        """
        数値情報をもとに、人間向けの文章コメントを生成する。
        「数値＋文章」の“文章”部分。
        """
        model = chosen.get("model", "unknown")
        score = float(chosen.get("score", 0.0))
        length = int(chosen.get("length", 0))
        status = chosen.get("status", "unknown")

        # スコア比較用に、他候補の最大スコアを調べておく
        other_scores = [
            float(c.get("score", 0.0))
            for c in candidates
            if c is not chosen
        ]
        max_other = max(other_scores) if other_scores else 0.0

        # モデル優先度
        pri = float(self.model_priority.get(model, 1.0))

        reasons: List[str] = []

        # status
        if status == "ok":
            reasons.append("応答が正常に生成されており")
        else:
            reasons.append("他候補に比べて相対的に条件が良かったため")

        # 長さ
        if length >= 900:
            reasons.append("物語として十分なボリュームがあり")
        elif length >= 400:
            reasons.append("適度な長さで内容がバランスよくまとまっており")
        elif length > 0:
            reasons.append("短めながら要点が整理されており")

        # 優先度
        if pri >= 6:
            reasons.append("Lyra-System 上で優先度の高いモデルに位置づけられているため")
        elif pri >= 4:
            reasons.append("安定性と表現力のバランスが良いモデルであるため")
        else:
            reasons.append("補助的なモデルながら内容が良好だったため")

        # 他候補との差
        if score > max_other:
            diff = score - max_other
            if diff >= 4.0:
                reasons.append("他の候補と比べて総合スコアが大きく上回っていました")
            elif diff >= 1.5:
                reasons.append("他の候補より一段高いスコアを記録していました")
            else:
                reasons.append("わずかにスコアが高く、総合的に優れていました")
        else:
            reasons.append("総合スコアが同程度であったため、わずかな差を評価しました")

        joined = "、".join(reasons)
        if not joined.endswith("。"):
            joined += "。"

        return f"{model} の出力が {joined}そのため、このラウンドでは {model} を採用しました。"
