    def _evaluate_pair(
        self,
        prompt: str,
        reply_a: str,
        reply_b: str,
        label_a: str,
        label_b: str,
    ) -> Dict[str, Any]:
        """
        実際に LLM に A/B 比較を依頼し、判定結果を dict で返します。
        """
        system_prompt = (
            "あなたは物語文の審査員です。\n"
            "同じプロンプトに対する 2 つの応答 A / B を比較し、"
            "どちらがより優れているかを判定してください。\n"
            "評価軸の例:\n"
            "  - 文体の自然さ\n"
            "  - 情景描写の豊かさ\n"
            "  - 感情表現の説得力\n"
            "  - これまでの文脈との整合性\n"
            "\n"
            "必ず次の JSON 形式だけを出力してください:\n"
            "{\n"
            '  \"winner\": \"A\" または \"B\",\n'
            '  \"score_diff\": 0.0〜1.0 の数値,\n'
            '  \"comment\": \"日本語で 1〜3 文の理由\"\n'
            "}\n"
        )

        user_content = (
            "【プロンプト】\n"
            f"{prompt}\n\n"
            "【応答A】\n"
            f"{reply_a}\n\n"
            "【応答B】\n"
            f"{reply_b}\n\n"
            "上記を比較し、指定された JSON 形式のみを出力してください。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        text, meta = call_with_fallback(
            messages=messages,
            temperature=0.0,
            max_tokens=300,
        )

        # 生テキストは必ず保持、JSON は成功したら別フィールドに
        result: Dict[str, Any] = {
            "winner": None,
            "score_diff": 0.0,
            "comment": "",
            "raw_text": text,   # ← LLMそのまま
            "raw_json": None,   # ← 解析に成功したら dict を入れる
            "route": meta.get("route"),
            "pair": {"A": label_a, "B": label_b},
        }

        parsed = self._safe_parse_json(text)

        if isinstance(parsed, dict):
            # パースに成功したら JSON の中身で上書き
            winner_raw = parsed.get("winner")
            if winner_raw == "A":
                result["winner"] = label_a
            elif winner_raw == "B":
                result["winner"] = label_b

            try:
                result["score_diff"] = float(parsed.get("score_diff", 0.0))
            except Exception:
                result["score_diff"] = 0.0

            comment = parsed.get("comment")
            if isinstance(comment, str):
                result["comment"] = comment.strip()

            # 解析済み JSON はここにだけ持つ
            result["raw_json"] = parsed

        # winner が決まらなかった場合のフォールバック
        if result["winner"] is None:
            result["winner"] = label_a
            result["score_diff"] = 0.0
            if not result["comment"]:
                result["comment"] = (
                    "JSON の解析に失敗したため、暫定的に A 側を選択しました。"
                )

        return result
