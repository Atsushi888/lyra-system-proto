from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import json
import glob

import streamlit as st


def _get_persona_json_dir() -> str:
    """
    PERSONA_JSON_DIR を secrets または環境変数から取得。
    見つからなければ 'actors/persona/persona_datas' を既定値とする。
    """
    base: Optional[str] = None

    # 1) st.secrets 優先
    try:
        secrets = st.secrets
        if isinstance(secrets, dict):
            base = secrets.get("PERSONA_JSON_DIR")
    except Exception:
        pass

    # 2) 環境変数
    if not base:
        base = os.getenv("PERSONA_JSON_DIR")

    # 3) デフォルト
    if not base:
        base = "actors/persona/persona_datas"

    return str(base)


def _load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def _save_json(path: str, data: Dict[str, Any]) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"JSON の保存に失敗しました: {e}")
        return False


def _ensure_list(val: Any) -> List[str]:
    """
    JSON 側が文字列・None などでも、とりあえず List[str] にして返す。
    """
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    # 改行区切りの文字列として扱う
    if isinstance(val, str):
        lines = [x.strip() for x in val.splitlines()]
        return [x for x in lines if x]
    return [str(val)]


def _list_to_multiline(items: List[str]) -> str:
    return "\n".join(items)


class PersonaEditor:
    """
    Persona JSON を編集するための UI コンポーネント（v2：項目全部盛り UI 版）。

    - PERSONA_JSON_DIR 配下の *.json から編集対象を選択
    - 主要フィールド（system_prompt / starter_hint / style_hint…）に加え、
      ペルソナ背景・性格・口調・感情パラメータを編集できる。
    - 現時点では「画面と JSON の往復」のみ。PersonaAI 等との連携は別工程。
    """

    def __init__(
        self,
        *,
        session_key: str = "persona_editor_json_state",
    ) -> None:
        self.session_key = session_key
        self.persona_dir = _get_persona_json_dir()

        # セッション内に編集中 JSON をキャッシュ
        if self.session_key not in st.session_state:
            st.session_state[self.session_key] = {
                "filename": "",
                "data": {},
            }

    # ------------------------------------------------------
    # セッションヘルパ
    # ------------------------------------------------------
    @property
    def state(self) -> Dict[str, Any]:
        return st.session_state[self.session_key]

    def _set_state(self, filename: str, data: Dict[str, Any]) -> None:
        st.session_state[self.session_key] = {
            "filename": filename,
            "data": data,
        }

    # ------------------------------------------------------
    def _list_json_files(self) -> List[str]:
        pattern = os.path.join(self.persona_dir, "*.json")
        files = sorted(glob.glob(pattern))
        return [os.path.basename(p) for p in files]

    # ------------------------------------------------------
    def render(self) -> None:
        st.markdown("## 🧬 Persona JSON エディタ（β・全部盛りUI）")

        st.caption(
            "PERSONA_JSON_DIR で指定されたディレクトリ配下の Persona JSON を選択し、"
            "主要フィールドや性格・口調・感情プロファイルを編集・保存できます。"
        )

        st.text_input(
            "Persona JSON directory",
            value=self.persona_dir,
            disabled=True,
        )

        json_files = self._list_json_files()
        if not json_files:
            st.error(
                "Persona JSON が見つかりませんでした。"
                f"ディレクトリ: `{self.persona_dir}`"
            )
            return

        # 現在選択中ファイル
        current_file = self.state.get("filename") or json_files[0]

        selected_file = st.selectbox(
            "編集する Persona JSON を選んでください:",
            options=json_files,
            index=max(
                json_files.index(current_file) if current_file in json_files else 0,
                0,
            ),
        )

        target_path = os.path.join(self.persona_dir, selected_file)
        st.text_input("対象ファイルパス", value=target_path, disabled=True)

        # 「JSON を再読み込み」ボタン
        if st.button("🔄 JSON を再読み込み", key="persona_reload_json"):
            data = _load_json(target_path)
            self._set_state(selected_file, data)
            st.success("JSON を再読み込みしました。")

        # 初期ロード
        if not self.state.get("data"):
            data = _load_json(target_path)
            self._set_state(selected_file, data)

        # 編集対象データ
        data: Dict[str, Any] = dict(self.state.get("data") or {})
        char_id = str(data.get("char_id", "floria_ja"))
        name = str(data.get("name", "フローリア"))

        st.markdown("---")
        st.markdown("### ✏️ 基本設定（編集可能）")
        col1, col2 = st.columns(2)
        with col1:
            char_id = st.text_input("キャラクターID", value=char_id)
        with col2:
            name = st.text_input("名前", value=name)

        data["char_id"] = char_id
        data["name"] = name

        # ====== system / starter / style ======
        with st.expander("system_prompt（ロール指示）", expanded=True):
            system_prompt = st.text_area(
                "system_prompt",
                value=str(data.get("system_prompt", "")),
                height=260,
            )
            data["system_prompt"] = system_prompt

        with st.expander("starter_hint（会話開始時ヒント）", expanded=False):
            starter_hint = st.text_area(
                "starter_hint",
                value=str(data.get("starter_hint", "")),
                height=160,
            )
            data["starter_hint"] = starter_hint

        with st.expander("style_hint（文体メモ）", expanded=False):
            style_hint = st.text_area(
                "style_hint",
                value=str(data.get("style_hint", "")),
                height=220,
            )
            data["style_hint"] = style_hint

        # ====== 背景・関係性 ======
        st.markdown("---")
        st.markdown("### 🌱 キャラクター背景・関係性")

        bio_lines = _ensure_list(data.get("bio"))
        bio_text = st.text_area(
            "bio（キャラの背景説明・1行1トピック）",
            value=_list_to_multiline(bio_lines),
            height=160,
            help="1行につき1つの記述として保存されます。",
        )
        data["bio"] = [x.strip() for x in bio_text.splitlines() if x.strip()]

        relationship = st.text_input(
            "relationship_to_player（プレイヤーとの関係性）",
            value=str(data.get("relationship_to_player", "")),
        )
        data["relationship_to_player"] = relationship

        goals_lines = _ensure_list(data.get("goals_and_dreams"))
        goals_text = st.text_area(
            "goals_and_dreams（目的・夢／1行1項目）",
            value=_list_to_multiline(goals_lines),
            height=140,
        )
        data["goals_and_dreams"] = [
            x.strip() for x in goals_text.splitlines() if x.strip()
        ]

        taboo_lines = _ensure_list(data.get("taboo_topics"))
        taboo_text = st.text_area(
            "taboo_topics（タブー／避けたい話題・行動）",
            value=_list_to_multiline(taboo_lines),
            height=120,
        )
        data["taboo_topics"] = [
            x.strip() for x in taboo_text.splitlines() if x.strip()
        ]

        # ====== 性格・行動パターン ======
        st.markdown("---")
        st.markdown("### 💡 性格・行動パターン")

        traits_pos_lines = _ensure_list(data.get("traits_positive"))
        traits_pos_text = st.text_area(
            "traits_positive（性格の長所／1行1項目）",
            value=_list_to_multiline(traits_pos_lines),
            height=120,
        )
        data["traits_positive"] = [
            x.strip() for x in traits_pos_text.splitlines() if x.strip()
        ]

        traits_neg_lines = _ensure_list(data.get("traits_negative"))
        traits_neg_text = st.text_area(
            "traits_negative（性格の短所／1行1項目）",
            value=_list_to_multiline(traits_neg_lines),
            height=120,
        )
        data["traits_negative"] = [
            x.strip() for x in traits_neg_text.splitlines() if x.strip()
        ]

        likes_lines = _ensure_list(data.get("likes"))
        likes_text = st.text_area(
            "likes（好きなもの／1行1項目）",
            value=_list_to_multiline(likes_lines),
            height=100,
        )
        data["likes"] = [x.strip() for x in likes_text.splitlines() if x.strip()]

        dislikes_lines = _ensure_list(data.get("dislikes"))
        dislikes_text = st.text_area(
            "dislikes（苦手なもの／1行1項目）",
            value=_list_to_multiline(dislikes_lines),
            height=100,
        )
        data["dislikes"] = [
            x.strip() for x in dislikes_text.splitlines() if x.strip()
        ]

        rules_lines = _ensure_list(data.get("behavioral_rules"))
        rules_text = st.text_area(
            "behavioral_rules（行動原則／NG行動など・1行1項目）",
            value=_list_to_multiline(rules_lines),
            height=140,
        )
        data["behavioral_rules"] = [
            x.strip() for x in rules_text.splitlines() if x.strip()
        ]

        # ====== 話し方・文体 ======
        st.markdown("---")
        st.markdown("### 🗣️ 話し方・口調")

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            first_person = st.text_input(
                "first_person（一人称）",
                value=str(data.get("first_person", "わたし")),
            )
        with col_p2:
            second_person = st.text_input(
                "second_person（二人称の基本呼び・例: あなた）",
                value=str(data.get("second_person", "あなた")),
            )
        with col_p3:
            politeness_level = st.selectbox(
                "politeness_level（丁寧さ）",
                options=["polite", "casual", "mix"],
                index=["polite", "casual", "mix"].index(
                    str(data.get("politeness_level", "mix"))
                    if str(data.get("politeness_level", "mix")) in ["polite", "casual", "mix"]
                    else "mix"
                ),
            )

        data["first_person"] = first_person
        data["second_person"] = second_person
        data["politeness_level"] = politeness_level

        speech_lines = _ensure_list(data.get("speech_patterns"))
        speech_text = st.text_area(
            "speech_patterns（口癖・言い回し・リズム／1行1項目）",
            value=_list_to_multiline(speech_lines),
            height=160,
        )
        data["speech_patterns"] = [
            x.strip() for x in speech_text.splitlines() if x.strip()
        ]

        # ====== 感情プロファイル ======
        st.markdown("---")
        st.markdown("### 💓 感情プロファイル（EmotionAI 連携用の種）")

        emo: Dict[str, Any] = data.get("emotional_tendencies") or {}
        if not isinstance(emo, dict):
            emo = {}

        def _get_emo(key: str, default: float) -> float:
            try:
                return float(emo.get(key, default))
            except Exception:
                return default

        st.caption("0.0 〜 1.0 の範囲で、おおよその傾向値を指定します。")

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            baseline_affection = st.slider(
                "baseline_affection（デフォルト好意度）",
                0.0,
                1.0,
                _get_emo("baseline_affection", 0.8),
                step=0.05,
            )
            baseline_arousal = st.slider(
                "baseline_arousal（デフォルトの感情の高まり）",
                0.0,
                1.0,
                _get_emo("baseline_arousal", 0.4),
                step=0.05,
            )
            shyness = st.slider(
                "shyness（照れやすさ）",
                0.0,
                1.0,
                _get_emo("shyness", 0.7),
                step=0.05,
            )
        with col_e2:
            jealousy = st.slider(
                "jealousy（嫉妬しやすさ）",
                0.0,
                1.0,
                _get_emo("jealousy", 0.6),
                step=0.05,
            )
            anger_threshold = st.slider(
                "anger_threshold（怒りが表面化するまでのしきい値／高いほど怒りにくい）",
                0.0,
                1.0,
                _get_emo("anger_threshold", 0.9),
                step=0.05,
            )

        data["emotional_tendencies"] = {
            "baseline_affection": baseline_affection,
            "baseline_arousal": baseline_arousal,
            "shyness": shyness,
            "jealousy": jealousy,
            "anger_threshold": anger_threshold,
        }

        # ====== 開発者向けメタ情報（任意） ======
        st.markdown("---")
        st.markdown("### 🧾 開発者メモ（任意）")

        notes = str(data.get("dev_notes", ""))
        notes = st.text_area(
            "dev_notes（開発メモ・将来の自分へのメッセージなど）",
            value=notes,
            height=120,
        )
        data["dev_notes"] = notes

        # JSON 生データ確認用
        with st.expander("JSON 全体プレビュー（読み取り専用）", expanded=False):
            st.json(data)

        # ====== 保存ボタン ======
        st.markdown("---")
        save_col1, save_col2 = st.columns([1, 1])
        with save_col1:
            if st.button("💾 この JSON に保存する", type="primary"):
                if _save_json(target_path, data):
                    self._set_state(selected_file, data)
                    st.success("JSON を保存しました。")
        with save_col2:
            if st.button("❌ 変更を破棄して再読み込み"):
                fresh = _load_json(target_path)
                self._set_state(selected_file, fresh)
                st.info("変更を破棄し、ファイル内容を再読み込みしました。")
