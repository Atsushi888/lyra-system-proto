from __future__ import annotations

from typing import Optional, Dict, Any, List
import os
import json

import streamlit as st


# Persona JSON を格納しているディレクトリ
# 例）.streamlit/secrets.toml に
# PERSONA_JSON_DIR = "actors/persona_datas"
# のように定義しておく。
PERSONA_JSON_DIR: str = st.secrets.get("PERSONA_JSON_DIR", "actors/persona_datas")


class PersonaEditor:
    """
    Persona JSON を読み書きするためのエディタ（β）。

    機能:
      - PERSONA_JSON_DIR 配下の *.json を列挙
      - 選択した JSON を読み込み、主要フィールドをフォームとして表示
      - 主要フィールド（char_id / name / system_prompt / starter_hint / style_hint）を編集
      - 「JSON に保存」ボタンで、同じファイルに書き戻し

    それ以外のフィールド（bio や traits など）は raw JSON ビューで確認のみ（現状）。
    """

    def __init__(
        self,
        *,
        base_dir: Optional[str] = None,
        session_prefix: str = "persona_editor",
    ) -> None:
        self.base_dir = base_dir or PERSONA_JSON_DIR
        self.session_prefix = session_prefix

    # -------------------------------------------------
    # 内部ヘルパ
    # -------------------------------------------------
    def _list_persona_files(self) -> List[str]:
        """base_dir 配下の *.json をアルファベット順に列挙。"""
        if not os.path.isdir(self.base_dir):
            return []

        files: List[str] = []
        for name in os.listdir(self.base_dir):
            path = os.path.join(self.base_dir, name)
            if os.path.isfile(path) and name.lower().endswith(".json"):
                files.append(name)

        files.sort()
        return files

    def _load_json(self, path: str) -> Optional[Dict[str, Any]]:
        """JSON を読み込んで dict を返す（失敗時は None）。"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"JSON の読み込みに失敗しました: {e}")
            return None

    def _save_json(self, path: str, data: Dict[str, Any]) -> bool:
        """JSON をファイルに保存する。成功すれば True。"""
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            st.error(f"JSON の保存に失敗しました: {e}")
            return False

    @staticmethod
    def _join_lines(value: Any) -> str:
        """
        system_prompt / style_hint などを表示用の1つのテキストにまとめる。
        - list[str] -> 行ごと
        - str       -> そのまま
        - それ以外 -> 空文字
        """
        if isinstance(value, list):
            return "\n".join(str(v) for v in value)
        if isinstance(value, str):
            return value
        return ""

    @staticmethod
    def _split_lines(text: str) -> List[str]:
        """
        text_area からの入力を list[str] に変換。
        - 空行＆前後空白は削る
        """
        lines = [line.strip() for line in text.splitlines()]
        return [line for line in lines if line]

    # -------------------------------------------------
    # メイン描画
    # -------------------------------------------------
    def render(self) -> None:
        st.markdown("## 🧬 Persona JSON エディタ（β）")
        st.caption(
            "PERSONA_JSON_DIR で指定されたディレクトリ配下の Persona JSON を選択し、"
            "主要フィールドを編集・保存できます。"
        )

        st.code(f"Persona JSON directory: {self.base_dir}", language="text")

        # ディレクトリ存在チェック
        if not os.path.isdir(self.base_dir):
            st.error(f"ディレクトリが存在しません: {self.base_dir}")
            st.info(
                "`.streamlit/secrets.toml` の PERSONA_JSON_DIR を確認するか、"
                "デフォルトの `actors/persona_datas` に JSON を配置してください。"
            )
            return

        # *.json 列挙
        files = self._list_persona_files()
        if not files:
            st.warning("指定ディレクトリに Persona JSON (*.json) が見つかりませんでした。")
            return

        # ファイル選択
        selected = st.selectbox(
            "編集する Persona JSON を選んでください：",
            files,
            key=f"{self.session_prefix}_file_select",
        )

        full_path = os.path.join(self.base_dir, selected)
        st.text(f"対象ファイル: {full_path}")

        data = self._load_json(full_path)
        if data is None:
            return

        # 主要フィールドを抽出
        char_id_default = str(data.get("char_id", ""))
        name_default = str(data.get("name", ""))

        system_prompt_text = self._join_lines(data.get("system_prompt"))
        starter_hint_text = str(data.get("starter_hint", ""))
        style_hint_text = self._join_lines(data.get("style_hint"))

        st.markdown("---")

        # ===== 編集フォーム =====
        st.markdown("### ✏️ 基本設定（編集可能）")

        form_key = f"{self.session_prefix}_form_{selected}"
        with st.form(key=form_key):
            # キーはファイル名込みでユニークにしておく
            key_prefix = f"{self.session_prefix}_{selected}_"

            col1, col2 = st.columns(2)
            with col1:
                char_id = st.text_input(
                    "キャラクターID",
                    value=char_id_default,
                    key=key_prefix + "char_id",
                )
            with col2:
                name = st.text_input(
                    "名前",
                    value=name_default,
                    key=key_prefix + "name",
                )

            with st.expander("system_prompt（ロール指示）", expanded=True):
                system_prompt_edit = st.text_area(
                    "system_prompt（1行1要素として扱われます）",
                    value=system_prompt_text,
                    height=260,
                    key=key_prefix + "system_prompt",
                )

            with st.expander("starter_hint（会話開始ヒント）", expanded=False):
                starter_hint_edit = st.text_area(
                    "starter_hint",
                    value=starter_hint_text,
                    height=160,
                    key=key_prefix + "starter_hint",
                )

            with st.expander("style_hint（文体メモ）", expanded=True):
                style_hint_edit = st.text_area(
                    "style_hint（1行1要素として扱われます）",
                    value=style_hint_text,
                    height=220,
                    key=key_prefix + "style_hint",
                )

            st.markdown("---")
            save_clicked = st.form_submit_button("💾 この JSON に保存する")

        # ===== 保存処理 =====
        if save_clicked:
            new_data: Dict[str, Any] = dict(data)  # 既存フィールドは維持したまま更新

            new_data["char_id"] = char_id.strip()
            new_data["name"] = name.strip()

            new_data["system_prompt"] = self._split_lines(system_prompt_edit)
            new_data["starter_hint"] = starter_hint_edit.strip()
            new_data["style_hint"] = self._split_lines(style_hint_edit)

            if self._save_json(full_path, new_data):
                st.success("JSON を保存しました。")
                # 保存直後の内容を確認用に表示
                with st.expander("保存後 JSON の内容（確認用）", expanded=False):
                    st.json(new_data)
            else:
                st.error("JSON の保存に失敗しました。")

        # ===== raw JSON ビュー =====
        st.markdown("---")
        st.markdown("### 📦 Raw JSON ビュー（参考）")
        st.caption(
            "上のフォームでは扱っていない追加フィールド（bio / traits など）は、"
            "ここで直接 JSON として確認できます。"
        )
        st.json(data)
