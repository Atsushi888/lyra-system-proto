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
    gpt-5.1 のように message.content が「パーツのリスト」になっている形にも対応する。
    """
    text = ""
    usage_dict: Optional[Dict[str, Any]] = None

    try:
        choices = getattr(completion, "choices", None) or []
        if choices:
            msg = getattr(choices[0], "message", None)
            if msg is not None:
                content_obj = getattr(msg, "content", "") or ""

                # 1) すでに str の場合
                if isinstance(content_obj, str):
                    text = content_obj

                # 2) list[part, part, ...] の場合（gpt-5.1 など）
                elif isinstance(content_obj, list):
                    parts: List[str] = []
                    for p in content_obj:
                        # p.text を優先して取り出す
                        t = getattr(p, "text", None)
                        if t is None and isinstance(p, dict):
                            t = p.get("text")
                        if t is None:
                            t = str(p)
                        parts.append(t)
                    text = "".join(parts)

                # 3) その他の型はとりあえず文字列化
                else:
                    text = str(content_obj)
    except Exception:
        logger.exception("OpenAI completion parse error")
        text = ""

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
    - 呼び出し側が max_tokens を指定してきた場合：
        * max_completion_tokens が未指定ならコピー
        * その後 max_tokens キーは削除
    こうしておけば、gpt-5.1 のように max_tokens を拒否するモデルでも安全。
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

    - name:  論理モデル名（"gpt51", "grok", "gemini", "hermes", "llama_unc" など）
    - call:  (messages, **kwargs) -> (text, usage_dict or None)
    """

    name: str = ""
    # モデルごとの「推奨トークン長」
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
            raise RuntimeError(f"{self.name}: OpenAI API キーが設定されていません。")

        # Lyra 内部用キーワードは削除
        for k in ("mode", "judge_mode"):
            kwargs.pop(k, None)

        # gpt-5.1 系だけは推論トークン抑えめにしておく
        if self.name == "gpt51" and "reasoning" not in kwargs:
            kwargs["reasoning"] = {"effort": "low"}

        # デフォルト max_tokens
        if (
            self.TARGET_TOKENS is not None
            and "max_tokens" not in kwargs
            and "max_completion_tokens" not in kwargs
        ):
            kwargs["max_completion_tokens"] = int(self.TARGET_TOKENS)

        _normalize_max_tokens(kwargs)

        last_exc: Optional[Exception] = None

        def _bump_max_tokens() -> None:
            # length 打ち切り時に少しずつ増やす（上限 2048 程度）
            inc = 160
            if "max_completion_tokens" in kwargs:
                cur = int(kwargs["max_completion_tokens"])
                kwargs["max_completion_tokens"] = min(cur + inc, 2048)
            elif "max_tokens" in kwargs:
                cur = int(kwargs["max_tokens"])
                kwargs["max_tokens"] = min(cur + inc, 2048)
            else:
                kwargs["max_completion_tokens"] = 512

        for attempt in range(3):  # 最大 2 回リトライ（合計 3 回）
            try:
                completion = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    **kwargs,
                )
                text, usage = _split_text_and_usage_from_openai_completion(completion)

                choices = getattr(completion, "choices", None) or []
                finish_reason = (
                    getattr(choices[0], "finish_reason", "") if choices else ""
                )

                # ちゃんとテキストが取れていればそれで終了
                if text.strip():
                    return text, usage

                # gpt-5.1 特有：「reasoning だけで打ち切り」パターンを救済
                if finish_reason == "length" and attempt < 2:
                    _bump_max_tokens()
                    continue  # もう一度だけトライ

                # それ以外の「空文字」は諦めてそのまま返す
                return text, usage

            except Exception as e:
                last_exc = e
                logger.exception(
                    "%s: OpenAI call failed (attempt=%s)", self.name, attempt + 1
                )

        raise RuntimeError(f"{self.name}: OpenAI call failed after retry: {last_exc}")


class GPT4oAdapter(OpenAIChatAdapter):
    """
    gpt-4o-mini 用アダプタ。
    （今後 AnswerTalker 側では使わず、別用途で使う想定）
    """

    def __init__(self) -> None:
        super().__init__(name="gpt4o", model_id="gpt-4o-mini")
        # 必要なら TARGET_TOKENS を設定（いまは制御しない）
        self.TARGET_TOKENS = None


class GPT51Adapter(OpenAIChatAdapter):
    """
    gpt-5.1 用アダプタ。
    出力が長くなりがちなので、やや短めの max_completion_tokens をデフォルトにする。
    """

    def __init__(self) -> None:
        super().__init__(name="gpt51", model_id="gpt-5.1")
        # 感情豊かだがクドくなりすぎない程度
        self.TARGET_TOKENS = 220


# ============================================================
# OpenRouter（Hermes / Llama Uncensored）Adapters
# ============================================================

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
HERMES_MODEL_OLD_DEFAULT = os.getenv(
    "OPENROUTER_HERMES_MODEL",
    # ここは環境変数で上書きされる前提（デフォルトは旧安定版名）
    "nousresearch/hermes-2-pro-mistral",
)
LLAMA_UNC_MODEL_DEFAULT = os.getenv(
    "OPENROUTER_LLAMA_UNC_MODEL",
    # ★必要に応じて OpenRouter の実際のモデルIDに合わせて変更すること
    "nousresearch/llama-3.1-70b-instruct:uncensored",
)


class HermesBaseAdapter(BaseLLMAdapter):
    """
    Hermes 系モデルの共通基底アダプタ。
    """

    def __init__(self, name: str, model_id: str) -> None:
        self.name = name
        self.model_id = model_id
        self._endpoint = OPENROUTER_BASE_URL.rstrip("/") + "/chat/completions"
        self._api_key = os.getenv("OPENROUTER_API_KEY", "")

    def call(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not self._api_key:
            raise RuntimeError("OPENROUTER_API_KEY が設定されていません。")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
        }

        # max_tokens 未指定なら TARGET_TOKENS を使う
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


class HermesOldAdapter(HermesBaseAdapter):
    """
    旧 Hermes（nousresearch/hermes-2-pro-*）用アダプタ。
    通常運用はこちらを "hermes" として使う想定。
    """

    def __init__(self) -> None:
        super().__init__(
            name="hermes",
            model_id=HERMES_MODEL_OLD_DEFAULT,
        )
        # erotic モードでのやり取りにちょうど良い程度
        self.TARGET_TOKENS = 260


class HermesNewAdapter(HermesBaseAdapter):
    """
    新 Hermes（3/4 系）用アダプタ。
    当面はテスト用："hermes_new" 名義で扱う。
    """

    def __init__(self) -> None:
        super().__init__(
            name="hermes_new",
            model_id="nousresearch/hermes-4-70b",
        )
        # テスト用途なので少し長めでも許容
        self.TARGET_TOKENS = 320


class LlamaUncensoredAdapter(HermesBaseAdapter):
    """
    Llama 3.1 70B Uncensored 用アダプタ。

    - OpenRouter で NousResearch 系の Uncensored モデルを叩く想定
    - 実際のモデルIDは OPENROUTER_LLAMA_UNC_MODEL 環境変数で差し替え可能
    """

    def __init__(self) -> None:
        super().__init__(
            name="llama_unc",
            model_id=LLAMA_UNC_MODEL_DEFAULT,
        )
        # 甘め長文をある程度許容
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
        # GPT より少し長めでもよいくらい
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

        # max_tokens 未指定なら TARGET_TOKENS を使う
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
        # Flash らしさは保ちつつ、短すぎない程度
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

        # OpenAI / 内部用のパラメータは全部捨てる
        for k in ("mode", "judge_mode", "max_tokens", "max_completion_tokens"):
            kwargs.pop(k, None)

        params: Dict[str, Any] = {
            "contents": self._to_gemini_contents(messages),
        }

        # generationConfig 未指定なら TARGET_TOKENS を反映
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
