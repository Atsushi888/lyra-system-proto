# actors/judge_ai3.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import random


class JudgeAI3:
    """
    複数 LLM の回答候補（models）から、
    「どのモデルのテキストを採用するか」を決める審判クラス。

    v0.4 方針:
      - models: { model_name: {"status": "ok", "text": "...", ...}, ... }
      - 「好みの長さモード」に近いテキストを高評価
        - preferred_length_mode: "auto" / "short" / "normal" / "long" / "story"
      - auto/normal モードでは従来どおり「ユーザー発話長さ」も参照
      - 将来的に内容評価ロジックを足していけるように、スコア算出はメソッド分離

    run() の戻り値:
      {
        "status": "ok" | "error",
        "error": str,
        "chosen_model": str,
        "chosen_text": str,
        "reason": str,
        "candidates": [
          {
            "name": str,
            "score": float,
            "length": int,
            "text": str,
            "status": str,
            "details": {
              "target_length": int,
              "length_mode": str,
              "length_score": float,
            },
          },
          ...
        ],
      }
    """

    def __init__(self, mode: str = "normal") -> None:
        # judge の「内容モード」（ツン期などで使う）とは別物
        self.mode = (mode or "normal").lower()

    def set_mode(self, mode: str) -> None:
        self.mode = (mode or "normal").lower()

    # ==========================================================
    # メインエントリ
    # ==========================================================
    def run(
        self,
        models: Dict[str, Any],
        user_text: str = "",
        preferred_length_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        models: ModelsAI2.collect() の結果（llm_meta["models"]）を想定。
        user_text: プレイヤーの直近発話（任意）。渡されなければ長さ150相当で計算。
        preferred_length_mode:
            UserSettings などから渡される「発話長さモード」。
            "auto" / "short" / "normal" / "long" / "story"
        """
        if not isinstance(models, dict) or not models:
            return {
                "status": "error",
                "error": "no_models",
                "chosen_model": "",
                "chosen_text": "",
                "reason": "models is empty or not a dict",
                "candidates": [],
            }

        length_mode = (preferred_length_mode or "auto").lower()
        user_len = len(user_text or "")
        target_len = self._calc_preferred_length(
            user_len=user_len,
            length_mode=length_mode,
        )

        candidates: List[Dict[str, Any]] = []

        for name, info in models.items():
            if not isinstance(info, dict):
                continue

            status = str(info.get("status") or "unknown")
            text = (info.get("text") or "").strip()
            length = len(text)

            if not text or status != "ok":
                score = -1.0
                length_score = 0.0
            else:
                length_score = self._score_length(
                    length=length,
                    target_length=target_len,
                )
                # 将来ここに「内容スコア」などを加算していく
                score = length_score

            candidates.append(
                {
                    "name": name,
                    "score": float(score),
                    "length": length,
                    "text": text,
                    "status": status,
                    "details": {
                        "target_length": target_len,
                        "length_mode": length_mode,
                        "length_score": float(length_score),
                    },
                }
            )

        if not candidates:
            return {
                "status": "error",
                "error": "no_candidates_built",
                "chosen_model": "",
                "chosen_text": "",
                "reason": "no candidates could be constructed from models",
                "candidates": [],
            }

        best = max(candidates, key=lambda c: c["score"])

        if best["score"] < 0:
            return {
                "status": "error",
                "error": "no_usable_candidate",
                "chosen_model": "",
                "chosen_text": "",
                "reason": "all candidates had non-positive scores",
                "candidates": candidates,
            }

        chosen_model = best["name"]
        chosen_text = best["text"]
        chosen_len = best["length"]

        reason = (
            f"preferred_length={target_len}, "
            f"length_mode={length_mode}, "
            f"user_length={user_len}, "
            f"chosen_model={chosen_model}, "
            f"chosen_length={chosen_len}"
        )

        return {
            "status": "ok",
            "error": "",
            "chosen_model": chosen_model,
            "chosen_text": chosen_text,
            "reason": reason,
            "candidates": candidates,
        }

    # ==========================================================
    # ターゲット長計算
    # ==========================================================
    def _calc_preferred_length(self, *, user_len: int, length_mode: str) -> int:
        """
        プレイヤーの発話長さ + length_mode から、
        このターンで「好み」とする回答長を決める。

        length_mode:
          - "auto"   … 旧仕様そのまま（気分屋モードあり）
          - "normal" … 旧仕様ベース（気分屋オフ / ノイズ弱め）
          - "short"  … かなり短め固定
          - "long"   … 会話中心のロング
          - "story"  … ミニシーン級のロング

        単位は「文字数」を想定。
        """

        m = (length_mode or "auto").lower()

        # user_len が 0 のときは「中庸な長さ」とみなす
        if user_len <= 0:
            user_len = 150

        # 0〜1 に正規化（300字以上は1扱い）
        u = max(0.0, min(1.0, user_len / 300.0))

        # ------------------------------------------------------
        # 明示モード（short / long / story / normal）
        # ------------------------------------------------------
        if m == "short":
            # だいたい 60〜140 文字くらい
            base = 90
            noise = random.randint(-30, 30)
            target = base + noise
            return max(40, target)

        if m == "long":
            # だいたい 220〜340 文字くらい
            base = 280
            noise = random.randint(-60, 60)
            target = base + noise
            return max(160, target)

        if m == "story":
            # だいたい 380〜520 文字くらい（しっかり長め）
            base = 450
            noise = random.randint(-80, 80)
            target = base + noise
            return max(280, target)

        if m == "normal":
            # 旧仕様をベースに、極端モードをオフにした版
            target_long = 260   # u ≒ 0.0 のとき
            target_short = 120  # u ≒ 1.0 のとき
            base_target = int(round(target_long * (1.0 - u) + target_short * u))

            # ゆらぎレンジ（auto より少し小さめ）
            max_noise = int(30 * (1.0 - u) + 8 * u)
            noise = random.randint(-max_noise, max_noise)
            target = base_target + noise
            return max(60, target)

        # ------------------------------------------------------
        # auto … 従来どおり「気分屋」含むモード
        # ------------------------------------------------------

        # 🎲 たまに極端モード
        r = random.random()

        # 1/20 ≒ 0.05 で「超短い」モード
        if r < 0.05:
            # 40〜80文字くらいの超ショート
            target = random.randint(40, 80)
            return target

        # 次の 1/20 で「超長い」モード（合計 1/10 で極端になる）
        if r < 0.10:
            # 260〜420文字くらいのロングモード
            target = random.randint(260, 420)
            return target

        # 通常モード（旧仕様）
        target_long = 260   # u ≒ 0.0 のとき
        target_short = 120  # u ≒ 1.0 のとき
        base_target = int(round(target_long * (1.0 - u) + target_short * u))

        max_noise = int(40 * (1.0 - u) + 10 * u)
        noise = random.randint(-max_noise, max_noise)

        target = base_target + noise
        return max(60, target)

    # ==========================================================
    # 長さスコア（0.0〜1.0）
    # ==========================================================
    @staticmethod
    def _score_length(*, length: int, target_length: int) -> float:
        """
        回答の文字数が「ターゲット長」にどれだけ近いかを 0.0〜1.0 で返す。

        diff が target と同じくらい離れていれば 0、
        ぴったりなら 1、というシンプルな線形スコア。
        """
        if length <= 0 or target_length <= 0:
            return 0.0

        diff = abs(length - target_length)
        rel = diff / float(target_length)

        score = 1.0 - rel  # diff == target_length → 0.0
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        return score
