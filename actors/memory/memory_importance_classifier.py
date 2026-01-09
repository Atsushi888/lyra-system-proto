# actors/memory/memory_importance_classifier.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from llm.llm_manager import LLMManager


ChatReturn = Union[str, Dict[str, Any], Tuple[Any, ...]]


class MemoryImportanceClassifier:
    """
    記憶イベントの「重要度(1-4)」「要約」「タグ」を、他AI（単発）に判定させる分類器。
    ※合議はしない。

    目的:
      - 「外泊」など、世界変化(importance=5)ではないが記憶に残すべき事象を拾う
      - AI Manager (enabled_models) の設定に従い、利用可能なモデルのみを使う

    出力（JSON）:
      {
        "importance": 1..4,
        "summary": "...",
        "tags": ["...","..."]
      }
    """

    def __init__(
        self,
        *,
        persona_id: str = "default",
        preferred_model: str = "gpt52",
        llm_manager: Optional[LLMManager] = None,
    ) -> None:
        self.persona_id = str(persona_id or "default")
        self.preferred_model = str(preferred_model or "gpt52")
        self._llm = llm_manager or LLMManager.get_or_create(persona_id=self.persona_id)

    def classify(
        self,
        *,
        messages: List[Dict[str, str]],
        final_reply: str,
        event_keywords_hit: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
          {
            "status": "ok"|"error",
            "importance": int(1..4),
            "summary": str,
            "tags": List[str],
            "model": str,
            "raw_text": str,
            "error": Optional[str],
          }
        """
        model = self._pick_model()

        system_prompt = (
            "You are a strict JSON generator for a memory system.\n"
            "Return ONLY valid JSON with keys: importance, summary, tags.\n"
            "Rules:\n"
            "- importance must be an integer 1..4\n"
            "- summary must be a short Japanese sentence (<= 80 chars preferred)\n"
            "- tags must be a JSON array of short Japanese tags (1..5 items)\n"
            "- Do not include any extra keys.\n"
            "- Do not wrap in markdown.\n"
        )

        user_prompt = self._build_prompt(messages, final_reply, event_keywords_hit or [])

        try:
            completion: ChatReturn = self._llm.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=220,
            )
            raw_text = self._normalize_text(completion)
            parsed = self._safe_parse_json(raw_text)

            importance = self._clamp_importance(parsed.get("importance"))
            summary = str(parsed.get("summary") or "").strip()
            tags = parsed.get("tags")
            tags_list = [str(x).strip() for x in tags] if isinstance(tags, list) else []

            if not summary:
                summary = self._fallback_summary(messages, final_reply)
            if not tags_list:
                tags_list = self._fallback_tags(event_keywords_hit or [])

            return {
                "status": "ok",
                "importance": importance,
                "summary": summary,
                "tags": tags_list[:5],
                "model": model,
                "raw_text": raw_text,
                "error": None,
            }

        except Exception as e:
            return {
                "status": "error",
                "importance": 3 if (event_keywords_hit or []) else 2,
                "summary": self._fallback_summary(messages, final_reply),
                "tags": self._fallback_tags(event_keywords_hit or []),
                "model": model,
                "raw_text": "",
                "error": str(e),
            }

    # ---------------- internal ----------------

    def _pick_model(self) -> str:
        """
        AI Manager の enabled / has_key を尊重する。
        - get_available_models() を正とする
        - enabled=False は使わない
        - has_key=False は使わない（もし付いていれば）
        """
        props = self._llm.get_available_models() or {}

        candidates = [
            self.preferred_model,
            "gpt52",
            "gpt51",
            "gpt4o",
            "gemini",
            "grok",
        ]
        for m in candidates:
            p = props.get(m)
            if not isinstance(p, dict):
                continue
            if p.get("enabled", True) is False:
                continue
            # get_available_models が has_key を付ける実装なら尊重
            if p.get("has_key", True) is False:
                continue
            return m

        # どうしても見つからない場合は preferred_model を返す（呼び出し側で例外→フォールバックされる）
        return self.preferred_model

    @staticmethod
    def _normalize_text(completion: ChatReturn) -> str:
        if isinstance(completion, dict):
            return str(completion.get("text") or completion.get("content") or completion.get("message") or "")
        if isinstance(completion, (tuple, list)):
            return "" if not completion else str(completion[0] or "")
        return "" if completion is None else str(completion)

    @staticmethod
    def _safe_parse_json(text: str) -> Dict[str, Any]:
        s = (text or "").strip()
        if not s:
            return {}

        # 余計な前後テキストが混ざった場合、最初の { から最後の } までを切る
        if "{" in s and "}" in s:
            s2 = s[s.find("{") : s.rfind("}") + 1]
        else:
            s2 = s

        try:
            obj = json.loads(s2)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _clamp_importance(v: Any) -> int:
        try:
            n = int(v)
        except Exception:
            n = 3
        if n < 1:
            return 1
        if n > 4:
            return 4
        return n

    @staticmethod
    def _fallback_summary(messages: List[Dict[str, str]], final_reply: str) -> str:
        # 最後の user を優先
        last_user = ""
        for m in reversed(list(messages or [])):
            if isinstance(m, dict) and m.get("role") == "user":
                last_user = str(m.get("content") or "").strip()
                if last_user:
                    break

        base = (last_user or "").strip() or (final_reply or "").strip()
        if not base:
            return "（記憶要約を作成できませんでした）"
        return base[:80] + ("…" if len(base) > 80 else "")

    @staticmethod
    def _fallback_tags(hit: List[str]) -> List[str]:
        tags = []
        if hit:
            tags.append("イベント")
        # よくある拾い方
        for k in hit[:3]:
            if k and k not in tags:
                tags.append(k)
        return tags or ["イベント"]

    @staticmethod
    def _build_prompt(
        messages: List[Dict[str, str]],
        final_reply: str,
        hit_keywords: List[str],
    ) -> str:
        lines: List[str] = []

        # 直近中心で十分（長すぎると判定がブレる）
        for m in (messages or [])[-10:]:
            role = m.get("role")
            if role not in ("user", "assistant"):
                continue
            content = (m.get("content") or "").strip()
            if content:
                lines.append(f"<{role.upper()}> {content}")

        fr = (final_reply or "").strip()
        if fr:
            lines.append(f"<ASSISTANT_FINAL> {fr}")

        hit = ", ".join([str(x) for x in hit_keywords if str(x).strip()]) if hit_keywords else "none"

        return (
            "You will judge if this conversation turn should be stored as memory.\n"
            "Important: This is NOT a world-level irreversible change. Use 1..4.\n\n"
            "Keyword hits (may be relevant): " + hit + "\n\n"
            "Return JSON only.\n"
            "Guideline for importance:\n"
            "1: trivial / small talk\n"
            "2: minor but potentially relevant later\n"
            "3: notable personal event / relationship-relevant\n"
            "4: major turning point but not world-irreversible\n\n"
            "Conversation:\n"
            + "\n".join(lines)
        )
