# council/council_manager.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

import streamlit as st


Speaker = Literal["player", "floria", "system"]
Mode = Literal["idle", "ongoing", "ended"]


@dataclass
class CouncilState:
    round: int = 0
    speaker: Speaker = "player"
    mode: Mode = "idle"
    log: List[Dict[str, Any]] = field(default_factory=list)
    # input はロジックでは使わないが、将来用に残しておく
    input: str = ""


class CouncilManager:
    """
    会談システムの中核ロジック。
    - Streamlit の session_state をラップして状態を保持
    - 画面描画もここでまとめて行う
    """

    # ★ 空文字は禁止。必ずプレフィックスを付ける
    SESSION_PREFIX = "council_"

    def __init__(self) -> None:
        self.state = st.session_state
        self._ensure_state()

    # ===== 状態管理ヘルパ =====
    def _key(self, name: str) -> str:
        """session_state / widget 用のキーを一元生成"""
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

        st.markdown("### プレイヤー入力")

        if mode != "ongoing":
            st.caption("（今はプレイヤーのターンではありません。会談を開始してから話してね）")
            return

        if speaker != "player":
            st.caption("（現在の話者はプレイヤーではありません。ターン待ちです）")
            return

        # --- 入力欄 ---
        # ログの長さを key に混ぜることで、送信のたびに新しい widget key になり、
        # テキストエリアの内容が自動的にクリアされる。
        input_key = self._key(f"input_{len(log)}")

        user_text: str = st.text_area(
            "あなたの発言：",
            key=input_key,
            placeholder="ここにフローリアや他の登場人物への発言を書いてください。",
        )

        col_input_btn, _ = st.columns([1, 3])
        with col_input_btn:
            if st.button("送信", key=self._key("send")):
                text = (user_text or "").strip()
                if text:
                    self._append_log("player", text)
                    # ★ widget の key が次回は変わるので、明示的にクリアする必要なし
                st.rerun()
