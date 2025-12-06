# actors/persona/affection_prompt_utils.py
from __future__ import annotations

from typing import Any, Dict

from actors.emotion_ai import EmotionResult
from actors.emotion_levels import affection_to_level


def build_system_prompt_with_affection(
    persona: Any,
    base_system_prompt: str,
    emotion: EmotionResult | None,
    doki_power: float = 0.0,
) -> str:
    """
    （旧API）Persona + EmotionResult + doki_power から、
    「好感度レベルに応じたデレ指示入り system_prompt」を組み立てる。

    ※ 現在は build_emotion_header() と組み合わせて使う想定。
    """
    system_prompt = base_system_prompt or ""

    if emotion is None:
        return system_prompt

    base_aff = float(getattr(emotion, "affection", 0.0) or 0.0)
    try:
        dp = float(doki_power)
    except Exception:
        dp = 0.0

    score = base_aff + dp
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0

    hint = ""
    if hasattr(persona, "build_affection_hint_from_score"):
        try:
            hint = persona.build_affection_hint_from_score(score)
        except Exception:
            hint = ""

    if hint:
        system_prompt = system_prompt + "\n\n" + hint

    return system_prompt


# ============================================================
# 新API：EmotionResult / doki_level / world_state → Emotionヘッダ
# ============================================================

def _clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _clamp_int(value: int, lo: int, hi: int) -> int:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def build_emotion_header(
    persona: Any,
    emotion: EmotionResult | None,
    world_state: Dict[str, Any] | None = None,
    scene_emotion: Dict[str, Any] | None = None,
) -> str:
    """
    EmotionResult + world_state などから、
    LLM 用の「感情・関係性ヘッダテキスト」を構築する。

    ここで doki_level を 0〜4 段階で扱う：

        0 … ほぼフラット
        1 … ちょっとトキメキ（片想い〜好意）
        2 … かなり意識してる（付き合い始め）
        3 … 人の目も気にならない（ゾッコン）
        4 … エクストリーム：結婚前提でベタ惚れ

    Persona 側が以下のメソッドを持っている場合はフックする：

        - get_emotion_profile(self) -> dict | None
            例：
                {
                    "affection_gain": 1.2,   # affection を少し盛る
                    "doki_bias": 1.0,        # doki_level を +1 段階甘く読む
                }

        - build_emotion_header_hint(self, emotion, world_state, scene_emotion) -> str
            → 独自にヘッダを全部書きたい場合はここで完結させてよい。
    """
    if emotion is None:
        return ""

    world_state = world_state or {}
    scene_emotion = scene_emotion or {}

    # 1) Persona 側の完全オーバーライドがあればそれを優先
    if hasattr(persona, "build_emotion_header_hint"):
        try:
            custom = persona.build_emotion_header_hint(
                emotion=emotion,
                world_state=world_state,
                scene_emotion=scene_emotion,
            )
            if isinstance(custom, str) and custom.strip():
                return custom.strip()
        except Exception:
            # 失敗したら共通ロジックにフォールバック
            pass

    # 2) Persona 側のプロファイルで係数を少しだけ調整
    aff_gain = 1.0
    doki_bias = 0.0

    if hasattr(persona, "get_emotion_profile"):
        try:
            prof = persona.get_emotion_profile() or {}
            aff_gain = float(prof.get("affection_gain", 1.0) or 1.0)
            doki_bias = float(prof.get("doki_bias", 0.0) or 0.0)
        except Exception:
            pass

    # 3) affection_with_doki をベースに好感度レベルを決める
    base_aff = float(getattr(emotion, "affection", 0.0) or 0.0)
    aff_with_doki = float(
        getattr(emotion, "affection_with_doki", base_aff) or base_aff
    )
    aff = _clamp(aff_with_doki * aff_gain, 0.0, 1.0)

    # 4) doki_level 0〜4 を取得してバイアス適用
    try:
        dl_raw = int(getattr(emotion, "doki_level", 0) or 0)
    except Exception:
        dl_raw = 0

    dl = int(round(dl_raw + doki_bias))
    dl = _clamp_int(dl, 0, 4)

    # 5) affection のゾーン（low/mid/high/extreme）
    aff_zone = affection_to_level(aff)

    # 6) doki_level に応じたニュアンス説明
    if dl == 0:
        doki_desc = (
            "いまは気持ちは比較的フラットですが、"
            "基本的な信頼や好意は保ちながら、落ち着いた口調で話してください。"
        )
    elif dl == 1:
        doki_desc = (
            "相手に対して『ちょっとトキメキ』を感じています。"
            "まだ自分の恋心をはっきり自覚していないものの、"
            "ふとした拍子に胸が高鳴る様子を、セリフや仕草に少しだけ滲ませてください。"
        )
    elif dl == 2:
        doki_desc = (
            "すでに『付き合い始め』レベルで相手を意識しています。"
            "二人きりの時間を嬉しそうに受け止め、遠慮がちな甘えや、"
            "ささやかなスキンシップを自然に織り交ぜてください。"
        )
    elif dl == 3:
        doki_desc = (
            "『人の目もあまり気にならないゾッコン状態』です。"
            "相手の言動ひとつひとつに強く心が揺れ、"
            "将来を本気でイメージしながら、甘く親密な距離感で接してください。"
        )
    else:  # dl == 4
        doki_desc = (
            "『エクストリーム：結婚前提でベタ惚れ』の段階です。"
            "相手を人生の伴侶候補として深く信頼し、"
            "ときどき将来の家庭や家族の話題にも触れながら、"
            "穏やかで包み込むような甘さと独占欲をにじませてください。"
        )

    # 7) affection ゾーンの短い説明
    if aff_zone == "low":
        aff_desc = (
            "まだ距離感はやや遠めで、憧れや緊張が入り混じった状態です。"
            "礼儀正しさを保ちつつ、ときどき素直な感情がこぼれる程度に留めてください。"
        )
    elif aff_zone == "mid":
        aff_desc = (
            "かなり打ち解けており、素直な好意や甘えが見え始めています。"
            "冗談や軽いツッコミを交えながら、親しみのあるコウハイらしい距離感で話してください。"
        )
    elif aff_zone == "high":
        aff_desc = (
            "先輩への恋心をはっきり自覚しており、ほぼ両想いに近い甘さになっています。"
            "二人の思い出や将来の約束に触れつつ、照れと幸福感を混ぜた口調で話してください。"
        )
    else:  # "extreme"
        aff_desc = (
            "すでに深く想い合っており、心のなかでは結婚や将来の生活まで見据えています。"
            "安心感と信頼をベースに、ときどき真剣な言葉や誓いのような台詞を織り交ぜてください。"
        )

    # 8) world_state / scene_emotion はここでは軽く言及に留める（必要なら拡張）
    location = world_state.get("location_name") or world_state.get("player_location")
    time_slot = world_state.get("time_slot") or world_state.get("time_of_day")

    scene_hint_parts: list[str] = []
    if location:
        scene_hint_parts.append(f"いま二人は『{location}』付近にいます。")
    if time_slot:
        scene_hint_parts.append(f"時間帯は『{time_slot}』頃です。")

    scene_hint = " ".join(scene_hint_parts).strip()

    header_lines: list[str] = []
    header_lines.append("【感情・関係性プロファイル】")
    header_lines.append(
        f"- 実効好感度（affection_with_doki）: {aff:.2f} "
        f"(zone={aff_zone}, doki_level={dl})"
    )
    if scene_hint:
        header_lines.append(f"- 環境: {scene_hint}")

    header_lines.append("")
    header_lines.append("【口調・距離感のガイドライン】")
    header_lines.append("1) 好感度ゾーンに基づくベースの方針：")
    header_lines.append(aff_desc)
    header_lines.append("")
    header_lines.append("2) ドキドキ💓レベルに基づく追加ニュアンス：")
    header_lines.append(doki_desc)
    header_lines.append("")
    header_lines.append(
        "上記をふまえ、キャラクターとして一貫性のある口調と感情表現で返答してください。"
        "ただし不自然に過剰なベタベタさではなく、その場の状況に合った自然な甘さと距離感を大切にしてください。"
    )

    return "\n".join(header_lines)
