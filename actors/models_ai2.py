class ModelsAI2:
    ...

    def collect(
        self,
        messages: List[Dict[str, str]],
        *,
        system_prompt: Optional[str] = None,
        mode_current: str = "normal",
        emotion_override: Optional[Dict[str, Any]] = None,
        reply_length_mode: str = "auto",
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        for model_name in self.enabled_models:
            try:
                completion = self.llm_manager.chat(
                    model=model_name,
                    messages=messages,
                    system_prompt=system_prompt,  # ★ ここが本命
                )

                norm = self._normalize_completion(completion)
                results[model_name] = {
                    "status": "ok",
                    "text": norm["text"],
                    "raw": norm["raw"],
                    "usage": norm["usage"],
                    "error": None,
                    "mode_current": mode_current,
                    "emotion_override": emotion_override,
                    "reply_length_mode": reply_length_mode,
                }

            except Exception as e:
                results[model_name] = {
                    "status": "error",
                    "text": "",
                    "raw": None,
                    "usage": None,
                    "error": str(e),
                    "mode_current": mode_current,
                    "emotion_override": emotion_override,
                    "reply_length_mode": reply_length_mode,
                }

        return results
