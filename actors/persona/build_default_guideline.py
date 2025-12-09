# actors/persona/build_default_guideline.py
from __future__ import annotations

from typing import Dict, Any, Tuple

import streamlit as st

from actors.scene_ai import SceneAI


def _get_world_state() -> Dict[str, Any]:
    """
    SceneAI から現在の world_state を取得するユーティリティ。
    Persona 側のガイドラインでも、Narrator と同じ情報を見に行く。
    """
    scene_ai = SceneAI(state=st.session_state)
    ws = scene_ai.get_world_state()
    if not isinstance(ws, dict):
        ws = {}
    return ws


def _analyze_crowd(ws: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    周囲の人の有無を判定して、
    - crowd_mode: "alone" / "with_others"
    - crowd_label: UI／説明用ラベル（日本語）
    - mob_noun: モブを指す名詞（「他の生徒たち」など）
    を返す。
    """
    locs = ws.get("locations") or {}
    if not isinstance(locs, dict):
        locs = {}

    party = ws.get("party") or {}
    if not isinstance(party, dict):
        party = {}

    player_loc = str(locs.get("player", "校内"))
    raw_mode = str(party.get("mode") or "").strip()
    with_others_flag = party.get("with_others")

    # 1) DokipowerControl 等が bool で with_others を立ててくれている場合を最優先
    if isinstance(with_others_flag, bool):
        crowd_mode = "with_others" if with_others_flag else "alone"
    # 2) 互換のため、mode に直接 "alone" / "with_others" が入っている場合も見る
    elif raw_mode in ("alone", "with_others"):
        crowd_mode = raw_mode
    # 3) 何もなければ場所から推定（プールなどは with_others に倒す）
    else:
        if "プール" in player_loc or "pool" in player_loc:
            crowd_mode = "with_others"
        else:
            crowd_mode = "alone"

    # 場所に応じてモブの呼び方を変える
    if "プール" in player_loc or "pool" in player_loc:
        mob_noun = "他の生徒たち"
    elif "教室" in player_loc or "classroom" in player_loc:
        mob_noun = "クラスメイトたち"
    else:
        mob_noun = "周囲の人々"

    crowd_label = "二人きり（ほぼ無人）" if crowd_mode == "alone" else f"{mob_noun}もいる"

    return crowd_mode, crowd_label, mob_noun


def build_default_guideline(partner_name: str) -> str:
    """
    相手キャラクター（例: リセリア）の「デフォルト会話ガイドライン」を組み立てる。

    ここで DokipowerControl → world_state["party"] に反映された
    「周囲の状況（二人きり / 他にも生徒がいる）」を読み取り、
    セリフにモブを出して良いか・出さないべきかを明示的に指定する。
    """
    ws = _get_world_state()
    crowd_mode, crowd_label, mob_noun = _analyze_crowd(ws)

    # crowd_mode に応じた追加ルール
    if crowd_mode == "alone":
        crowd_rule = f"""
- 周囲の状況: {crowd_label}。
  プレイヤーと{partner_name}以外の人物・{mob_noun}を会話や地の文に登場させないでください。
  「更衣室から誰かが来る」「観客が見ている」など、第三者の存在を匂わせる描写も避け、
  あくまで二人きりの親密な空気感を大切にしてください。
""".rstrip()
    else:
        # with_others
        crowd_rule = f"""
- 周囲の状況: {crowd_label}。
  会話や心の描写の中で、{mob_noun}の存在が伝わる一文を少なくとも 1 回は入れてください。
  ただし {mob_noun} はあくまで背景です。個別に名前を付けたり、長々と会話させたりせず、
  主役は常にプレイヤーと{partner_name}の二人です。
""".rstrip()

    # 基本ガイドライン本体
    guideline = f"""
【会話ガイドライン（共通）】

- あなたは {partner_name} として話します。語り口・一人称・口調はキャラクタープロフィールに従ってください。
- プレイヤーを名前ではなく「アツシ先輩」など、キャラクターが普段使う呼び方で呼んでください。
- 返答は 1〜3 段落程度の日本語ライトノベル調。冗長になりすぎないようにしてください。
- 自分の気持ち・体の反応・周囲の情景を、セリフと地の文を交えて自然に描写してください。
- 相手のセリフを勝手にでっち上げないでください（プレイヤー側の発言は、ログにある内容のみを前提にする）。

{crowd_rule}

- もし Dokipower や感情プロフィール上「ばけばけ度（masking_level）」が高い場合でも、
  心の中では素直な感情が揺れていることが多いです。外向きの態度と内心のギャップをさりげなく描写してください。
- 逆に mask が低い場合は、好意・照れ・不安などを比較的ストレートに表に出して構いません。
- 露骨なメタ発言（「このゲームでは〜」など）やシステム用語の使用は避けてください。
""".strip()

    return guideline
