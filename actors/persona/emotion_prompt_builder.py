# actors/persona/emotion_prompt_builder.py
from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_emotion_based_system_prompt(
    *,
    base_system_prompt: str,
    emotion_override: Optional[Dict[str, Any]] = None,
    mode_current: str = "normal",
) -> str:
    """
    - 既存の system prompt（キャラクター説明など）に
      affection / doki 情報をヘッダとして付け足した完全版 system_prompt を返す。
    - 「何も情報がない」ケースでも安全に動作する。
    """
    emotion_override = emotion_override or {}
    world_state = emotion_override.get("world_state") or {}
    scene_emotion = emotion_override.get("scene_emotion") or {}
    emotion = emotion_override.get("emotion") or {}

    # affection は doki 補正後を優先
    affection = float(
        emotion.get("affection_with_doki", emotion.get("affection", 0.0)) or 0.0
    )
    doki_power = int(emotion.get("doki_power", 0) or 0)
    doki_level = int(emotion.get("doki_level", 0) or 0)

    # zone は現状はプレースホルダ（将来チューニング用）
    zone = "auto"

    # doki_level 0–4 を言語化
    if doki_level >= 4:
        rel_stage = "エクストリーム：結婚を前提にベタ惚れしているレベル。未来を強く意識している。"
    elif doki_level == 3:
        rel_stage = "ディープ：強い信頼と恋情が混ざり合い、人目をあまり気にせず甘えたりできる段階。"
    elif doki_level == 2:
        rel_stage = "ミドル：付き合い始めのような甘い関係。お互いに好意を自覚している。"
    elif doki_level == 1:
        rel_stage = "ライト：ほのかな恋心。まだ自分の気持ちを探っている段階。"
    else:
        rel_stage = "フラット：恋愛感情としてはまだ芽生え始め〜穏やかな友好的関係。"

    # world_state から舞台情報を抽出（あれば）
    loc_player = (world_state.get("locations") or {}).get("player")
    time_info = world_state.get("time") or {}
    time_slot = time_info.get("slot")
    time_str = time_info.get("time_str")

    location_lines: List[str] = []
    if loc_player:
        location_lines.append(f"- 現在の舞台は「{loc_player}」。")
    if time_slot or time_str:
        ts = f"{time_slot} / {time_str}" if time_slot and time_str else (time_slot or time_str)
        location_lines.append(f"- 時間帯は「{ts}」。")

    # 口調ガイドラインは doki_level に応じて少し変える
    guideline_lines: List[str] = []

    if doki_level >= 4:
        guideline_lines.extend(
            [
                "1) 結婚を前提にした深い信頼と愛情を前提として、将来への期待がにじむトーンで話してください。",
                "2) さりげないスキンシップや将来の生活を匂わせる表現を、セリフの中に1つ以上含めてください。",
                "3) 『ずっとそばにいたい』『本気で大事にしたい』と伝わるニュアンスを自然な描写で入れてください。",
            ]
        )
    elif doki_level == 3:
        guideline_lines.extend(
            [
                "1) 強い好意と信頼が伝わる、親密で少し独占欲のにじむトーンで話してください。",
                "2) 距離が近いことや触れそうな距離感を意識した描写を、会話の中にさりげなく混ぜてください。",
                "3) 相手の体調や気持ちを気遣う言葉を交えつつ、『あなたが大切』というニュアンスを含めてください。",
            ]
        )
    elif doki_level == 2:
        guideline_lines.extend(
            [
                "1) 付き合い始めのような甘さと緊張感のバランスを意識しながら話してください。",
                "2) 視線・手の位置・距離感など、少しドキドキしそうな要素を描写に含めてください。",
                "3) からかい半分・本気半分のような、照れ混じりのセリフを入れても構いません。",
            ]
        )
    elif doki_level == 1:
        guideline_lines.extend(
            [
                "1) 基本は丁寧で礼儀正しいが、ときどき素直な感情がこぼれるトーンで話してください。",
                "2) 相手を意識して少しだけ言葉に詰まったり、照れがにじむ描写を入れてください。",
            ]
        )
    else:
        guideline_lines.extend(
            [
                "1) まだ大きな恋愛感情としては動いていないが、好感や信頼は感じられるフラットなトーンで話してください。",
                "2) 落ち着いた会話の中に、相手を気遣う一言をさりげなく入れてください。",
            ]
        )

    guideline_lines.append(
        "3) いずれの場合も、キャラクターとして一貫性のある口調と感情表現で返答してください。"
        " 不自然に過剰なベタベタさではなく、その場の状況に合った自然な甘さと距離感を大切にしてください。"
    )

    # ヘッダ本体
    header_lines: List[str] = []

    header_lines.append("[感情・関係性プロファイル]")
    header_lines.append(
        f"- 実効好感度 (affection_with_doki): {affection:.2f} (zone={zone}, doki_level={doki_level})"
    )
    header_lines.append(f"- 関係ステージ: {rel_stage}")
    if location_lines:
        header_lines.extend(location_lines)

    header_lines.append("")  # 空行
    header_lines.append("[口調・距離感のガイドライン]")
    guideline_lines = [line.rstrip() for line in guideline_lines]
    header_lines.extend(guideline_lines)

    header_block = "\n".join(header_lines)

    if base_system_prompt:
        new_system_prompt = base_system_prompt.rstrip() + "\n\n" + header_block + "\n"
    else:
        new_system_prompt = header_block + "\n"

    return new_system_prompt


def replace_system_prompt(
    messages: List[Dict[str, str]],
    new_system_prompt: str,
) -> List[Dict[str, str]]:
    """
    - messages 内の最初の system を new_system_prompt で置き換える。
    - system が無ければ、先頭に追加する。
    """
    new_messages = list(messages)
    system_index = None

    for idx, m in enumerate(new_messages):
        if m.get("role") == "system":
            system_index = idx
            break

    system_message = {
        "role": "system",
        "content": new_system_prompt,
    }

    if system_index is not None:
        new_messages[system_index] = system_message
    else:
        new_messages.insert(0, system_message)

    return new_messages
