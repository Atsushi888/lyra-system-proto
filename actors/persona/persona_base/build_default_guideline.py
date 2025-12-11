# actors/persona/persona_base/build_default_guideline.py
from __future__ import annotations

from typing import Any


def build_default_guideline(
    *,
    affection_with_doki: float,
    doki_level: int,
    mode_current: str,
) -> str:
    """
    emotion_profiles が未設定の場合に使うフォールバック用ガイドライン。

    かなり大ざっぱだが、最低限「好感度」「ドキドキ」「モード」に応じた
    距離感の目安を LLM に伝える。
    """
    lines: list[str] = []
    lines.append("[口調と距離感ガイドライン（デフォルト）]")

    # 好感度ゾーンざっくり
    if affection_with_doki < 0.15:
        lines.append("- まだそこまで親しくない段階。丁寧で少し距離のある話し方。")
    elif affection_with_doki < 0.4:
        lines.append("- ある程度打ち解けた相手。丁寧さは保ちつつ、親しみを込めた言い回しも許可。")
    elif affection_with_doki < 0.7:
        lines.append("- かなり仲の良い相手。感情表現もやや増やし、安心感のある距離感で接する。")
    else:
        lines.append("- 深く信頼している相手。素直な甘えや弱さを見せてもよい。")

    # doki_level のざっくり解釈
    if doki_level >= 4:
        lines.append("- 今回は胸の高鳴りがかなり強い。声や言い回しに少しだけ戸惑いや照れをにじませる。")
    elif doki_level >= 2:
        lines.append("- 少しドキドキしている。通常よりもわずかにトーンが柔らかくなる程度。")
    elif doki_level == 1:
        lines.append("- かすかな高揚感はあるが、基本は普段通り。")
    else:
        lines.append("- 特別な高揚感はないため、いつも通り落ち着いたトーンで。")

    # mode による微調整
    if mode_current == "erotic":
        lines.append(
            "- ただし、全体としては大人向けロマンス寄りの雰囲気をわずかに意識する。"
        )
    elif mode_current == "debate":
        lines.append("- 今回は議論寄りのモードなので、論理的・冷静な話し方を優先する。")

    return "\n".join(lines)
