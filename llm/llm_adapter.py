# llm/llm_adapter.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import logging

from openai import OpenAI as OpenAIClient
import requests

logger = logging.getLogger(__name__)


# ============================================================
# 共通ヘルパ
# ============================================================

def _split_text_and_usage_from_openai_completion(
    completion: Any,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    OpenAI ChatCompletion オブジェクトから text/usage を取り出す。
    gpt-5.1 で content 形式が変わっても、なるべく空文字にならないように
    フォールバックを多めに入れている。
    """
    text = ""
    usage_dict: Optional[Dict[str, Any]] = None

    # --- メイン経路 ---
    try:
        choices = getattr(completion, "choices", None) or []
        if choices:
            msg = getattr(choices[0], "message", None)
            if msg is not None:
                content = getattr(msg, "content", "") or ""
                # content が list の場合（マルチパート）、text 部をつないでしまう
                if isinstance(content, list):
                    parts: List[str] = []
                    for p in content:
                        if isinstance(p, dict):
                            parts.append(str(p.get("text", "")))
                        else:
                            parts.append(str(p))
                    content = "\n".join([s for s in parts if s])
                text = str(content)
    except Exception:
        logger.exception("OpenAI completion parse error (primary)")
        text = ""

    # --- dict へのフォールバック ---
    if not text:
        try:
            as_dict = completion.to_dict() if hasattr(completion, "to_dict") else None
            if isinstance(as_dict, dict):
                choices = as_dict.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    content = (
                        msg.get("content")
                        or msg.get("text")
                        or ""
                    )
                    text = str(content)
        except Exception:
            logger.exception("OpenAI completion parse error (fallback dict)")
            text = ""

    # --- 最終フォールバック：どうしても取れない場合は全体を文字列化 ---
    if not text:
        try:
            text = str(completion)
        except Exception:
            text = ""

    # usage
    usage = getattr(completion, "usage", None)
    if usage is not None:
        try:
            usage_dict = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        except Exception:
            usage_dict = None

    return text, usage_dict


def _split_text_and_usage_from_dict(
    data: Dict[str, Any],
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    OpenRouter / Grok / その他の dict レスポンスから text/usage を取り出す。
    """
    text = ""
    usage_dict: Optional[Dict[str, Any]] = None

    try:
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            text = msg.get("content", "") or ""
    except Exception:
        logger.exception("LLM dict response parse error")
        text = ""

    if "usage" in data:
        usage_dict = data.get("usage")

    return text, usage_dict


def _normalize_max_tokens(kwargs: Dict[str, Any]) -> None:
    """
    OpenAI の新 API では max_tokens ではなく max_completion_tokens を使う。
    """
    if "max_completion_tokens" in kwargs:
        kwargs.pop("max_tokens", None)
        return

    max_tokens = kwargs.pop("max_tokens", None)
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens


# ============================================================
# Base Adapter
# ============================================================

class BaseLLMAdapter:
    """
    各ベンダーごとの LLM 呼び出しをカプセル化する基底クラス。
    """

    name: str = ""
    TARGET_TOKENS: Optional[int] = None

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        raise NotImplementedError


# ============================================================
# OpenAI 系 Adapters (GPT-4o / GPT-5.1)
# ============================================================

class OpenAIChatAdapter(BaseLLMAdapter):
    """
    OpenAI Chat Completions 用の共通アダプタ。
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        env_key: str = "OPENAI_API_KEY",
    ) -> None:
        self.name = name
        self.model_id = model_id
        api_key = os.getenv(env_key, "")
        self._client: Optional[OpenAIClient] = (
            OpenAIClient(api_key=api_key) if api_key else None
        )

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if self._client is None:
            raise RuntimeError(
                f"{self.name}: OpenAI API キーが設定されていません。"
            )

        # 呼び出し側が max_* を指定していない場合、TARGET_TOKENS をデフォルトとして使用
        if (
            self.TARGET_TOKENS is not None
            and "max_tokens" not in kwargs
            and "max_completion_tokens" not in kwargs
        ):
            kwargs["max_completion_tokens"] = int(self.TARGET_TOKENS)

        _normalize_max_tokens(kwargs)

        completion = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            **kwargs,
        )
        return _split_text_and_usage_from_openai_completion(completion)


class GPT4oAdapter(OpenAIChatAdapter):
    def __init__(self) -> None:
        super().__init__(name="gpt4o", model_id="gpt-4o-mini")
        self.TARGET_TOKENS = None  # 今は使わない


class GPT51Adapter(OpenAIChatAdapter):
    """
    gpt-5.1 用アダプタ。
    出力が長くなりがちなので、やや短めの max_completion_tokens をデフォルトにする。
    """

    def __init__(self) -> None:
        super().__init__(name="gpt51", model_id="gpt-5.1")
        self.TARGET_TOKENS = 220


# ============================================================
# OpenRouter（Hermes）Adapters：旧版／新版
# ============================================================

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# 旧 Hermes のデフォルト ID（OpenRouter 用）
HERMES_MODEL_OLD_DEFAULT = os.getenv(
    "OPENROUTER_HERMES_MODEL",
    "nousresearch/hermes-2-pro-llama-3-8b",  # ← ここを変更
)


class HermesBaseAdapter(BaseLLMAdapter):
    """
    Hermes 系モデルの共通基底アダプタ。
    OpenRouter を OpenAI 互換エンドポイントとして叩く。
    """

    def __init__(self, name: str, model_id: str) -> None:
        self.name = name
        self.model_id = model_id
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if api_key:
            self._client: Optional[OpenAIClient] = OpenAIClient(
                api_key=api_key,
                base_url=OPENROUTER_BASE_URL,
            )
        else:
            self._client = None

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if self._client is None:
            raise RuntimeError("OPENROUTER_API_KEY が設定されていません。")

        # max_tokens 未指定なら TARGET_TOKENS を使う
        if self.TARGET_TOKENS is not None and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = int(self.TARGET_TOKENS)

        # OpenAI 互換なので max_tokens → max_completion_tokens 変換も使える
        _normalize_max_tokens(kwargs)

        completion = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            **kwargs,
        )
        return _split_text_and_usage_from_openai_completion(completion)


class HermesOldAdapter(HermesBaseAdapter):
    """
    旧 Hermes（nousresearch/hermes-2-pro-mistral）用アダプタ。
    通常運用はこちらを "hermes" として使う。
    """

    def __init__(self) -> None:
        super().__init__(
            name="hermes",
            model_id=HERMES_MODEL_OLD_DEFAULT,
        )
        self.TARGET_TOKENS = 260


class HermesNewAdapter(HermesBaseAdapter):
    """
    新 Hermes（3/4 系）用アダプタ。"hermes_new" 名義でテスト用。
    """

    def __init__(self) -> None:
        super().__init__(
            name="hermes_new",
            model_id="nousresearch/hermes-4-70b",
        )
        self.TARGET_TOKENS = 320


# ============================================================
# xAI（Grok）Adapter
# ============================================================

class GrokAdapter(BaseLLMAdapter):
    """
    xAI Grok 用アダプタ。
    デフォルトは短文傾向のため、やや長めの max_tokens を標準にする。
    """

    def __init__(
        self,
        name: str = "grok",
        model_id: str = "grok-2-latest",
    ) -> None:
        self.name = name
        self.model_id = model_id
        self._endpoint = "https://api.x.ai/v1/chat/completions"
        self._api_key = os.getenv("GROK_API_KEY", "")
        self.TARGET_TOKENS = 480

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not self._api_key:
            raise RuntimeError("GROK_API_KEY が設定されていません。")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
        }

        if self.TARGET_TOKENS is not None and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = int(self.TARGET_TOKENS)

        payload.update(kwargs)

        resp = requests.post(
            self._endpoint,
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return _split_text_and_usage_from_dict(data)


# ============================================================
# Google Gemini Adapter
# ============================================================

class GeminiAdapter(BaseLLMAdapter):
    """
    Google Gemini 2.0 用アダプタ。
    Flash 系は超短文になりやすいので、やや長めの maxOutputTokens を標準にする。
    """

    def __init__(
        self,
        name: str = "gemini",
        model_id: str = "gemini-2.0-flash-exp",
    ) -> None:
        self.name = name
        self.model_id = model_id
        self._endpoint = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_id}:generateContent"
        )
        self._api_key = os.getenv("GEMINI_API_KEY", "")
        self.TARGET_TOKENS = 400

    def _to_gemini_contents(
        self,
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        contents: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role", "user")
            text = m.get("content", "")
            contents.append(
                {
                    "role": "user" if role != "assistant" else "model",
                    "parts": [{"text": text}],
                }
            )
        return contents

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY が設定されていません。")

        params: Dict[str, Any] = {
            "contents": self._to_gemini_contents(messages),
        }

        if self.TARGET_TOKENS is not None and "generationConfig" not in kwargs:
            kwargs["generationConfig"] = {
                "maxOutputTokens": int(self.TARGET_TOKENS),
            }

        params.update(kwargs)

        resp = requests.post(
            self._endpoint,
            params={"key": self._api_key},
            json=params,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        text = ""
        try:
            cands = data.get("candidates") or []
            if cands:
                parts = cands[0].get("content", {}).get("parts", [])
                if parts:
                    text = parts[0].get("text", "") or ""
        except Exception:
            logger.exception("Gemini response parse error")

        return text, None
