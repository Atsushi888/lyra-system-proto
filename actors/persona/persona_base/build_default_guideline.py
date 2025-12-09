# actors/persona/build_default_guideline.py
from __future__ import annotations

from typing import Dict, Any, Optional


def _detect_others_present(world_state: Optional[Dict[str, Any]]) -> Optional[bool]:
    """
    world_state から「外野がいるかどうか」をできるだけ素直に推定する。

    優先順位:
      1) world_state["others_present"]   … Dokipower や他システムが直接立てたフラグ
      2) world_state["others_around"]    … 既存の互換フラグがあれば使う
      3) world_state["party_mode"] / world_state["party"]["mode"]
         - "alone" なら False（完全二人きり or 一人）
         - それ以外は None（ここでは判断保留）
    """
    ws = world_state or {}

    if "others_present" in ws:
        try:
            return bool(ws["others_present"])
        except Exception:
            pass

    if "others_around" in ws:
        try:
            return bool(ws["others_around"])
        except Exception:
            pass

    party_mode = ws.get("party_mode")
    if not party_mode:
        party = ws.get("party") or {}
        if isinstance(party, dict):
            party_mode = party.get("mode")

    if party_mode == "alone":
        # プレイヤー単独 or プレイヤー＋相手キャラのみ、という想定
        return False

    # ここでは「どっちとも言えない」扱いにしておく
    return None


def build_default_guideline(
    *,
    affection_with_doki: float,
    doki_level: int,
    mode_current: str,
    world_state: Optional[Dict[str, Any]] = None,
) -> str:
    """
    JSON に emotion_profiles が無いときのフォールバック用ガイドライン。

    ここで world_state を見て、
    - 二人きりか
    - 他の生徒（モブ）がいるか
    を system_prompt に明示する。
    """
    guideline_lines: list[str] = []
    guideline_lines.append("[口調・距離感のガイドライン]")

    # ----- ここで「モブの有無」を first line として明示 -----
    others_flag = _detect_others_present(world_state)

    if others_flag is False:
        # 完全に二人きり／外野なし
        guideline_lines.append(
            "0) 現在、この場にはプレイヤーと相手キャラクター以外の人物はいません。"
            "人目をあまり気にせず、二人きりの静かな環境として、"
            "素直な甘さや弱さをいつもより少しだけ強めに見せても構いません。"
        )
    elif others_flag is True:
        # 周囲に生徒・利用者がいる
        guideline_lines.append(
            "0) 現在、この場には他の生徒たちや通行人もいます。"
            "人前であることを自覚し、恋愛感情や独占欲をあからさまに見せすぎないようにしてください。"
            "外向きには一段階落ち着いたトーンを保ちつつ、視線や言い回し、ささやかな仕草にだけ感情をにじませてください。"
        )
    else:
        # 判定材料が無い場合は何も書かない（他の条件だけで判断させる）
        pass

    # ----- 以下は従来の doki_level ベースのデフォルト挙動 -----
    if doki_level >= 4:
        guideline_lines.extend(
            [
                "1) 結婚を前提にした深い信頼と愛情を前提として、将来への期待がにじむトーンで話してください。",
                "2) さりげないスキンシップや、一緒に暮らす将来を匂わせる表現をセリフの中に1つ以上含めてください。",
                "3) 『ずっとそばにいたい』『本気で大事にしたい』と伝わるニュアンスを、自然な描写で入れてください。",
            ]
        )
    elif doki_level == 3:
        guideline_lines.extend(
            [
                "1) 強い好意と信頼が伝わる、親密で少し独占欲のにじむトーンで話してください。",
                "2) 距離が近いことや袖をつまむなど、触れそうな距離感を意識した描写をセリフに混ぜてください。",
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
        "9) いずれの場合も、キャラクターとして一貫性のある口調と感情表現で返答してください。"
        " 不自然に過剰なベタベタさではなく、その場の状況（二人きりか、人前か）に合った自然な甘さと距離感を大切にしてください。"
    )

    return "\n".join(guideline_lines)
