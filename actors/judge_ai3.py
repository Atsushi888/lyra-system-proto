# actors/judge_ai3.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import random


class JudgeAI3:
    """
    複数 LLM の回答候補（models）から、
    「どのモデルのテキストを採用するか」を決める審判クラス。

    v0.5.0 方針:
      - models: { model_name: {"status": "ok", "text": "...", ...}, ... }
      - preferred_length_mode: "auto" / "short" / "normal" / "long" / "story"
      - priority: ["gpt52","gpt51",...] のような「モデル優先順位」
        → usable(=status ok & textあり) の中から、priorityの先頭を優先採用。
        → priorityで選べない場合のみ、従来の length_score にフォールバック。
    """

    def __init__(self, mode: str = "normal") -> None:
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
        priority: Optional[List[str]] = None,   # ★追加
    ) -> Dict[str, Any]:
        """
        models: ModelsAI2.collect() の結果（llm_meta["models"]）を想定。
        user_text: プレイヤーの直近発話（任意）。渡されなければ長さ150相当で計算。
        preferred_length_mode:
            UserSettings などから渡される「発話長さモード」。
            "auto" / "short" / "normal" / "long" / "story"
        priority:
            AIManager などから渡される「モデル優先順位」。
            例: ["gpt52","gpt51","grok","gemini","gpt4o"]
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

        prio: List[str] = [str(x) for x in (priority or []) if str(x).strip()]
        prio_rank: Dict[str, int] = {name: i for i, name in enumerate(prio)}

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
                        "priority_rank": prio_rank.get(name, 10**9),  # ★追加（UIデバッグ用）
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

        usable: List[Dict[str, Any]] = [
            c for c in candidates
            if c["status"] == "ok" and c["text"]
        ]

        if not usable:
            return {
                "status": "error",
                "error": "no_usable_candidate",
                "chosen_model": "",
                "chosen_text": "",
                "reason": "no candidates had status=ok and non-empty text",
                "candidates": candidates,
            }

        # ------------------------------------------------------
        # ① priority があるなら「優先順位の先頭」を採用（最重要）
        # ------------------------------------------------------
        if prio:
            by_name = {c["name"]: c for c in usable}
            chosen = None
            for name in prio:
                c = by_name.get(name)
                if c is not None:
                    chosen = c
                    break

            if chosen is not None:
                chosen_model = chosen["name"]
                chosen_text = chosen["text"]
                chosen_len = chosen["length"]
                selection_strategy = "priority_first"

                reason = (
                    f"selection={selection_strategy}, "
                    f"priority={prio}, "
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

        # ------------------------------------------------------
        # ② priority がない / 使える候補が priority に居ない → 従来通り
        # ------------------------------------------------------
        if length_mode == "story":
            best = max(usable, key=lambda c: c["length"])
            selection_strategy = "story_max_length"
        else:
            best = max(usable, key=lambda c: c["score"])
            selection_strategy = "length_score"

        chosen_model = best["name"]
        chosen_text = best["text"]
        chosen_len = best["length"]

        reason = (
            f"selection={selection_strategy}, "
            f"priority={prio if prio else 'none'}, "
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
        m = (length_mode or "auto").lower()

        if user_len <= 0:
            user_len = 150

        u = max(0.0, min(1.0, user_len / 300.0))

        if m == "short":
            base = 90
            noise = random.randint(-30, 30)
            target = base + noise
            return max(40, target)

        if m == "long":
            base = 280
            noise = random.randint(-60, 60)
            target = base + noise
            return max(160, target)

        if m == "story":
            base = 500
            noise = random.randint(-100, 100)
            target = base + noise
            return max(300, target)

        if m == "normal":
            target_long = 260
            target_short = 120
            base_target = int(round(target_long * (1.0 - u) + target_short * u))

            max_noise = int(30 * (1.0 - u) + 8 * u)
            noise = random.randint(-max_noise, max_noise)
            target = base_target + noise
            return max(60, target)

        r = random.random()

        if r < 0.05:
            target = random.randint(40, 80)
            return target

        if r < 0.10:
            target = random.randint(260, 420)
            return target

        target_long = 260
        target_short = 120
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
        if length <= 0 or target_length <= 0:
            return 0.0

        diff = abs(length - target_length)
        rel = diff / float(target_length)

        score = 1.0 - rel
        if score < 0.0:
            score = 0.0
        if score > 1.0:
            score = 1.0
        return score
