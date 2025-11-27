from __future__ import annotations
from typing import List, Dict, Any

import streamlit as st

from actors.actor import Actor
from personas.persona_floria_ja import Persona


def get_or_create_council_actor() -> Actor:
    """
    会談システム用の Actor を1つだけ生成・再利用する。
    LLMRouter はもう利用しない。AnswerTalker の内部で LLMManager を利用するため。
    """
    actor_key = "council_actor"

    if actor_key not in st.session_state:
        st.session_state[actor_key] = Actor(
            name="フローリア",
            persona=Persona(),
        )

    return st.session_state[actor_key]


class CouncilManager:
    """
    会談システムのロジック ＋ 画面描画（β）。
    - conversation_log: 会話の生ログ（プレイヤー/フローリア両方）
    - round は「発言の総数」として len(conversation_log) から毎回計算する
    """

    def __init__(self) -> None:
        # 会話ログ：List[{"role": "...", "content": "..."}]
        self.conversation_log: List[Dict[str, str]] = []

        # いまはフローリア AI だけ
        self.actors: Dict[str, Actor] = {
            "floria": Actor("フローリア", Persona())
            # session_state 共有したければ:
            # "floria": get_or_create_council_actor()
        }

        # 状態（round は持たず、都度計算）
        self.state: Dict[str, Any] = {
            "mode": "ongoing",
            "participants": ["player", "floria"],
            "last_speaker": None,
        }

    # ===== 内部ヘルパ =====
    def _append_log(self, role: str, content: str) -> None:
        """ログに 1 発言を追加。改行は <br> に変換して保存。"""
        safe = (content or "").replace("\n", "<br>")
        self.conversation_log.append({"role": role, "content": safe})
        self.state["last_speaker"] = role

    # ===== ロジック側 公開 API =====
    def reset(self) -> None:
        """会談を最初からやり直す。"""
        self.conversation_log.clear()
        self.state["mode"] = "ongoing"
        self.state["last_speaker"] = None

    def get_log(self) -> List[Dict[str, str]]:
        """会談ログのコピーを返す（表示用）。"""
        return list(self.conversation_log)

    def get_status(self) -> Dict[str, Any]:
        """
        サイドバー表示用のステータス。
        round は「これからプレイヤーが行う発言の番号」として計算する。
        """
        round_ = len(self.conversation_log) + 1
        return {
            "round": round_,
            "speaker": "player",  # いまは常にプレイヤーのターン開始とみなす
            "mode": self.state.get("mode", "ongoing"),
            "participants": self.state.get("participants", ["player", "floria"]),
            "last_speaker": self.state.get("last_speaker"),
        }

    def proceed(self, user_text: str) -> str:
        """
        プレイヤーの発言を受け取り、
        - ログに追加
        - フローリアに conversation_log 丸ごと渡して返事を生成
        - 返事もログに追加
        を行う。
        """
        # プレイヤー発言
        self._append_log("player", user_text)

        reply = ""
        actor = self.actors.get("floria")
        if actor is not None:
            reply = actor.speak(self.conversation_log)
            self._append_log("floria", reply)

        return reply

    # ===== 画面描画 =====
    def render(self) -> None:
        log = self.get_log()
        status = self.get_status()

        st.markdown("## 🗣️ 会談システム（Council Prototype）")
        st.caption("※ Actor ベースで AI と会話する会談システム（β）です。")

        # 上部コントロール
        col_left, col_right = st.columns([3, 1])
        with col_right:
            if st.button("🔁 リセット", key="council_reset"):
                self.reset()
                st.success("会談をリセットしました。")
                st.rerun()

        # ---- 会談ログ ----
        st.markdown("### 会談ログ")
        if not log:
            st.caption("（まだ会談は始まっていません。何か話しかけてみましょう）")
        else:
            for idx, entry in enumerate(log, start=1):
                role = entry.get("role", "")
                text = entry.get("content", "")
                if role == "player":
                    name = "プレイヤー"
                elif role == "floria":
                    name = "フローリア"
                else:
                    name = role or "？"

                st.markdown(f"**[{idx}] {name}**")
                st.markdown(text, unsafe_allow_html=True)
                st.markdown("---")

        # ---- サイドバー：会談ステータス ----
        with st.sidebar.expander("📊 会談ステータス", expanded=True):
            st.write(f"ラウンド: {status.get('round')}")
            st.write(f"話者: {status.get('speaker')}")
            st.write(f"モード: {status.get('mode')}")
            participants = status.get("participants") or []
            if participants:
                st.write("参加者: " + " / ".join(participants))
            last = status.get("last_speaker")
            if last:
                st.write(f"最後の話者: {last}")

        # ---- プレイヤー入力 ----
        st.markdown("### プレイヤー入力")

        # ラウンドごとに key を変えることで、送信後は別 key になり、入力欄は空になる
        round_no = int(status.get("round") or 1)
        input_key = f"council_user_input_r{round_no}"

        user_text = st.text_area(
            "あなたの発言：",
            key=input_key,
            placeholder="ここにフローリアへの発言を書いてください。",
        )

        raw_text = user_text or ""
        can_send = bool(raw_text.strip())  # ★ 空白のみなら送信不可

        send_col, _ = st.columns([1, 3])
        with send_col:
            send_clicked = st.button(
                "送信",
                key="council_send",
                disabled=not can_send,  # ★ 空欄なら押せない
            )

            if send_clicked and can_send:
                cleaned = raw_text.strip()

                # ★ この run の中で player → floria まで完結させる
                with st.spinner("フローリアは少し考えています…"):
                    self.proceed(cleaned)

                # proceed でログが増えたので、次の run では round が進む
                # → input_key も変わるので、入力欄は自動的に空の新しいボックスになる
                st.rerun()
