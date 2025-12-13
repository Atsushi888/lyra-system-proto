# llm2/llm_ai/llm_adapters/grok.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import logging
import requests

from llm2.llm_ai.llm_adapters.base import BaseLLMAdapter
from llm2.llm_ai.llm_adapters.utils import split_text_and_usage_from_dict

logger = logging.getLogger(__name__)


class GrokAdapter(BaseLLMAdapter):
    """
    xAI Grok 用アダプタ。

    変更点:
    - endpoint を環境変数で差し替え可能にする（Streamlit Cloudでも運用しやすい）
    - 404 のときだけ「末尾スラッシュ有り」も試す（xAI側のルーティング差異対策）
    """

    def __init__(
        self,
        *,
        name: str = "grok",
        model_id: str = "grok-2-latest",
        env_key: str = "GROK_API_KEY",
        endpoint_env: str = "GROK_API_ENDPOINT",
    ) -> None:
        self.name = name
        self.model_id = model_id

        # デフォルトは従来のまま。必要なら env で差し替え。
        self._endpoint = os.getenv(endpoint_env, "https://api.x.ai/v1/chat/completions").strip()
        self._api_key = os.getenv(env_key, "")

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

        def _post(url: str) -> requests.Response:
            return requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60,
            )

        try:
            resp = _post(self._endpoint)

            # ✅ 404 のときだけ「末尾スラッシュ」を試す（同一パスでも実際に差が出る環境がある）
            if resp.status_code == 404:
                alt = (self._endpoint.rstrip("/") + "/")
                if alt != self._endpoint:
                    resp = _post(alt)

            resp.raise_for_status()
            data = resp.json()

        except Exception as e:
            logger.exception("%s: Grok call failed", self.name)
            raise RuntimeError(f"{self.name}: Grok call failed: {e}")

        return split_text_and_usage_from_dict(data)
