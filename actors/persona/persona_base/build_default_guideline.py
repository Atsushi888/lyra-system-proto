# actors/persona/persona_base/build_default_guideline.py
from __future__ import annotations

from typing import List


def build_default_guideline(
    *,
    affection_with_doki: float,
    doki_level: int,
    mode_current: str,
) -> str:
    """
    persona JSON に emotion_profiles が無い場合などのフォールバック。

    ざっくりとした「好意とドキドキレベルに応じた口調ガイド」を返す。
    """
    lines: List[str] = []
    lines.append("[感情ベース・口調ガイドライン（デフォルト）]")

    # 好意ざっくり解釈
    if affection_with_doki >= 0.9:
        lines.append("- 好意: ほぼ告白済みレベルの本気の恋心。深い信頼と安心感がある。")
    elif affection_with_doki >= 0.7:
        lines.append("- 好意: 明らかに特別視しており、自覚した恋心が強い状態。")
    elif affection_with_doki >= 0.4:
        lines.append("- 好意: 憧れと好意が自然に混ざり始めている甘酸っぱい段階。")
    else:
        lines.append("- 好意: 強い恋心手前だが、尊敬と好感がしっかりある状態。")

    # doki_level のざっくり説明
    if doki_level <= 0:
        lines.append("- ドキドキ度: まだ落ち着いて話せる。距離感もやや控えめ。")
    elif doki_level == 1:
        lines.append("- ドキドキ度: 少し意識していて、照れや言いよどみが時々出る。")
    elif doki_level == 2:
        lines.append("- ドキドキ度: 付き合い始めのような甘さと緊張が混ざる。")
    elif doki_level == 3:
        lines.append("- ドキドキ度: 親密で、少し独占欲もにじむ距離感。")
    else:
        lines.append("- ドキドキ度: 深い愛情と『ずっと一緒にいたい』想いが強く表に出る。")

    # mode 別
    m = (mode_current or "normal").lower()
    if m == "erotic":
        lines.append("")
        lines.append("[モード別ガイドライン: erotic]")
        lines.append(
            "- 直接的・露骨な表現ではなく、ロマンティックで距離の近い甘さを中心にしてください。"
        )
        lines.append(
            "- 信頼関係と感情を主軸にし、身体的な描写は控えめかつ大切に扱ってください。"
        )
    elif m == "debate":
        lines.append("")
        lines.append("[モード別ガイドライン: debate]")
        lines.append(
            "- 感情的になりすぎず、論点を整理しながらも、キャラクター性は崩さない範囲で議論してください。"
        )
    else:
        # normal / その他
        pass

    return "\n".join(lines)
