# actors/composer_refiner.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable

from llm.llm_manager import LLMManager


def _call_llm_dynamic(
    llm_manager: LLMManager,
    model_name: str,
    prompt: str,
    *,
    temperature: float = 0.8,
    max_tokens: int = 900,
) -> str:
    """
    refiner_model (例: "gpt51") から LLMManager.call_model を経由して
    各モデルの Adapter を呼び出し、テキストを生成する。

    LLMManager のインターフェース:
      call_model(model_name, messages, temperature=..., max_tokens=..., ...)
    を前提とする。
    """
    if not isinstance(llm_manager, LLMManager):
        raise TypeError("llm_manager must be an instance of LLMManager")

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    raw = llm_manager.call_model(
        model_name=model_name,
        messages=messages,
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )

    # call_model の戻りは (text, usage) or str を想定
    if isinstance(raw, tuple):
        reply_text = raw[0]
    else:
        reply_text = raw

    if not isinstance(reply_text, str):
        reply_text = "" if reply_text is None else str(reply_text)

    return reply_text or ""


def build_refine_prompt(
    *,
    question: str,
    chosen_text: str,
    other_candidates: List[Dict[str, Any]],
    persona_style_hint: str,
) -> str:
    """
    Composer 用のリファインプロンプトを組み立てる。
    """
    others_snip_parts: List[str] = []
    for c in other_candidates:
        if not isinstance(c, dict):
            continue
        name = c.get("name") or c.get("model") or "unknown"
        txt = (c.get("text") or "").strip()
        if not txt:
            continue
        if len(txt) > 800:
            txt = txt[:800] + " ……（以下略）"
        others_snip_parts.append(f"【モデル {name} の参考回答】\n{txt}")

    others_block = (
        "\n\n".join(others_snip_parts)
        if others_snip_parts
        else "（他モデルの参考回答はありません）"
    )

    prompt = f"""
あなたは会話AIですが、次のペルソナとして振る舞います。

[ペルソナ・スタイル指示]
{persona_style_hint}

[ユーザーからの質問]
{question}

[ベースとなる回答（Judge によって選択済み）]
{chosen_text}

[他モデルからの参考回答（必要に応じて利用可）]
{others_block}

[あなたへの指示]
- ベースとなる回答を土台としつつ、他モデルの参考回答に含まれる「有用で矛盾しない情報」があれば、必要な範囲で統合してください。
- 同じ内容が重複している部分はうまくまとめ、矛盾がある場合は、より自然で一貫性があり、ユーザーにとって分かりやすい内容を優先してください。
- 回答は日本語で、ペルソナ・スタイル指示に従った口調・雰囲気で書いてください。
- 見出しや箇条書き、装飾記号は使わず、自然な文章の連なりとして答えてください。
- 回答はそのままユーザーに提示される最終メッセージになります。
"""
    return prompt.strip()


def make_default_refiner(
    *,
    llm_manager: LLMManager,
    temperature: float = 0.85,
    max_tokens: int = 900,
) -> Callable[
    [str, str, str, List[Dict[str, Any]], Dict[str, Any], Optional[str]],
    str,
]:
    """
    ComposerAI に注入するための「標準 refiner」を生成するファクトリ。

    旧仕様:
      make_default_refiner(router=..., model_props=..., ...)
    新仕様:
      make_default_refiner(llm_manager=..., ...)
    """

    def refiner(
        question: str,
        source_model: str,
        chosen_text: str,
        other_candidates: List[Dict[str, Any]],
        llm_meta: Dict[str, Any],
        refiner_model: Optional[str],
    ) -> str:
        persona_style_hint = ""
        if isinstance(llm_meta, dict):
            hint = llm_meta.get("persona_style_hint")
            if isinstance(hint, str):
                persona_style_hint = hint

        prompt = build_refine_prompt(
            question=question,
            chosen_text=chosen_text,
            other_candidates=other_candidates,
            persona_style_hint=persona_style_hint,
        )

        # 既定では gpt51 でリファイン
        model_name = refiner_model or "gpt51"

        refined_text = _call_llm_dynamic(
            llm_manager=llm_manager,
            model_name=model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return refined_text

    return refiner
