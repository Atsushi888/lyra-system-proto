# council/council_manager.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import streamlit as st

# ★ 追加：Actor クラス（登場人物）を受け取れるようにする
from actors.actor import Actor


Speaker = Literal["player", "floria", "system"]
Mode = Literal["idle", "ongoing", "ended"]


@dataclass
class CouncilState:
    round: int = 0
    speaker: Speaker = "player"
    mode: Mode = "idle"
    log: List[Dict[str, Any]] = field(default_factory=list)
    input: str = ""


class CouncilManager:
    """
    会談システムの中核ロジック。
    - Streamlit の session_state をラップして状態を保持
    - 画面描画もここでまとめて行う
    - ★ Actor クラス（player / floria / system など）を受け取って保持できる
    """

    SESSION_PREFIX = "council_"  # ★ 空文字は禁止。必ずプレフィックスを付ける

    def __init__(
        self,
        *,
        actors: Optional[Dict[Speaker, Actor]] = None,
    ) -> None:
        """
        actors:
            - "player" → プレイヤー Actor（任意）
            - "floria" → フローリア Actor（任意）
            - "system" → GM / フォルティナ用 Actor（任意）

        まだ現在は「誰が接続されているか」を保持して表示するだけ。
        実際に Actor.speak() を呼ぶのは、次のステップで実装する。
        """
        self.state = st.session_state
        self._ensure_state()

        # ★ Actor の接続情報（session_state ではなくインスタンス変数で持つ）
        #   → Streamlit は Python オブジェクトも普通に保持できるが、
        #     まずは「発注側」でだけ使うので、ここに抱えておく。
        self.actors: Dict[Speaker, Actor] = actors or {}

    # ===== 状態管理ヘルパ =====
    def _key(self, name: str) -> str:
        """session_state 用のキーを一元生成"""
        return f"{self.SESSION_PREFIX}{name}"

    def _ensure_state(self) -> None:
        """初期値がなければ作る"""
        defaults = CouncilState()
        for field_name, value in defaults.__dict__.items():
            key = self._key(field_name)
            if key not in self.state:
                self.state[key] = value

    def _get(self, name: str) -> Any:
        return self.state[self._key(name)]

    def _set(self, name: str, value: Any) -> None:
        self.state[self._key(name)] = value

    # ===== Actor 管理（発注先） =====

    def attach_actor(self, speaker: Speaker, actor: Actor) -> None:
        """
        後からでも Actor を差し込めるようにしておく。
        例:
            manager.attach_actor("floria", floria_actor)
        """
        self.actors[speaker] = actor

    def get_actor(self, speaker: Speaker) -> Optional[Actor]:
        """
        話者ラベル → Actor を取得。
        まだこのメソッドは使わないが、将来的に
        「floria のターンになったら floria_actor.speak(...) を呼ぶ」
        ための入り口になる。
        """
        return self.actors.get(speaker)

    # ===== API =====
    def reset(self) -> None:
        """会談をリセットして idle に戻す"""
        self._set("round", 0)
        self._set("speaker", "player")
        self._set("mode", "idle")
        self._set("log", [])
        self._set("input", "")

    def start(self) -> None:
        """会談開始"""
        self._set("round", 1)
        self._set("speaker", "player")
        self._set("mode", "ongoing")
        self._set("log", [])
        self._set("input", "")

    def _append_log(self, speaker: Speaker, text: str) -> None:
        log: List[Dict[str, Any]] = list(self._get("log"))
        log.append({"speaker": speaker, "text": text})
        self._set("log", log)

    # ===== メイン描画 =====
    def render(self) -> None:
        # ※毎回呼ばれるので保険として
        self._ensure_state()

        round_ = self._get("round")
        speaker: Speaker = self._get("speaker")
        mode: Mode = self._get("mode")
        log: List[Dict[str, Any]] = self._get("log")

        # --- ヘッダ ---
        st.markdown("## 🗣️ 会談システム（Council Prototype）")
        st.caption("※ ロジックとUIは CouncilManager に集約。ここから拡張していく。")

        # --- 上部コントロール ---
        col_left, col_right = st.columns([3, 1])
        with col_right:
            if st.button("🔁 会談リセット / 開始", key=self._key("reset_start")):
                # idle → start / それ以外 → reset & start
                self.start()
                st.rerun()

        # --- ログ表示 ---
        st.markdown("### 会談ログ")
        if not log:
            st.caption("（まだ会談が始まっていません。「会談リセット / 開始」でスタート）")
        else:
            for i, entry in enumerate(log, start=1):
                role = entry.get("speaker", "?")
                text = entry.get("text", "")
                if role == "player":
                    name = "プレイヤー"
                elif role == "floria":
                    name = "フローリア"
                else:
                    name = "システム"
                st.markdown(f"**[{i}] {name}**")
                st.markdown(text)
                st.markdown("---")

        # --- 右側ステータス ---
        with st.sidebar.expander("会談ステータス", expanded=True):
            st.write(f"ラウンド: {round_}")
            st.write(f"話者: {speaker}")
            st.write(f"モード: {mode}")

            # ★ どの話者に Actor がアタッチされているかも表示（デバッグ用）
            if self.actors:
                st.caption("接続中 Actor")
                for spk, act in self.actors.items():
                    st.write(f"- {spk}: {act.display_name} (id={act.id})")
            else:
                st.caption("接続中 Actor: なし")

        st.markdown("### プレイヤー入力")

        if mode != "ongoing":
            st.caption("（今はプレイヤーのターンではありません。会談を開始してから話してね）")
            return

        if speaker != "player":
            st.caption("（現在の話者はプレイヤーではありません。ターン待ちです）")
            return

        # --- 入力欄 ---
        input_key = self._key("input")
        user_text: str = st.text_area(
            "あなたの発言：",
            key=input_key,
            placeholder="ここにフローリアや他の登場人物への発言を書いてください。",
        )

        col_input_btn, _ = st.columns([1, 3])
        with col_input_btn:
            if st.button("送信", key=self._key("send")):
                text = (self.state.get(input_key) or "").strip()
                if text:
                    self._append_log("player", text)
                    # ★ 送信後に入力欄をクリア
                    self.state[input_key] = ""
                    # ★ 将来的にはここで：
                    #   - floria Actor に text を渡して次の返答をもらう
                    #   - speaker を "floria" に切り替える
                    #   などを行う予定。
                st.rerun()
