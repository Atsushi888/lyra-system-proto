# actors/answer_talker.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Mapping
import os
import traceback

import streamlit as st

from actors.models_ai2 import ModelsAI2
from actors.judge_ai3 import JudgeAI3
from actors.composer_ai import ComposerAI
from actors.memory_ai import MemoryAI
from actors.emotion_ai import EmotionAI, EmotionResult
from actors.emotion.emotion_models import EmotionModel
from actors.persona_ai import PersonaAI
from actors.scene_ai import SceneAI
from actors.mixer_ai import MixerAI
from actors.init_ai import InitAI
from llm.llm_manager import LLMManager

LYRA_DEBUG = os.getenv("LYRA_DEBUG", "0") == "1"


class AnswerTalker:
    """
    デバッグ強化版（改良）
    - 段階ごとに llm_meta に痕跡を残す
    - MemoryAI（新旧）互換で初期化
    - speak() の最後で MemoryAI.update_from_turn() を必ず走らせる

    重要な修正:
    - AIManager と同じ LLMManager(persona_id="default") を使い、enabled が必ず効くようにする
    - MemoryAI も default に統一（UI表示と保存先のズレを解消）
    """

    def __init__(
        self,
        persona: Any,
        llm_manager: Optional[LLMManager] = None,
        memory_model: str = "gpt52",
        state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.persona = persona
        self.state = state or st.session_state

        # Init
        InitAI.ensure_all(state=self.state, persona=self.persona)

        # -----------------------------
        # ★重要：LLMの統一persona_id
        # -----------------------------
        # AIManager が default を操作している前提で統一する
        self.runtime_persona_id = "default"

        # LLMManager（default）
        self.llm_manager = llm_manager or LLMManager.get_or_create(self.runtime_persona_id)

        # meta
        self.llm_meta: Dict[str, Any] = self.state.setdefault("llm_meta", {})
        self.llm_meta.setdefault("models", {})
        self.llm_meta.setdefault("errors", [])
        self.llm_meta.setdefault("stage", "init")

        # round_id 管理（AnswerTalker 側で確実に単調増加）
        if "round_id" not in self.state:
            self.state["round_id"] = 0

        # AIs
        # PersonaAI はキャラIDで良い（プロンプト置換など）
        persona_id_for_prompt = getattr(persona, "char_id", "default")
        self.persona_ai = PersonaAI(persona_id=persona_id_for_prompt)

        # ModelsAI2（enabled は内部で available_models.enabled を尊重）
        self.models_ai = ModelsAI2(self.llm_manager, persona=self.persona)

        # Emotion / Scene / Mixer
        # ※「gpt52だけ運用」したいなら、ここも gpt52 に寄せるのが安全
        self.emotion_ai = EmotionAI(self.llm_manager, model_name="gpt52")
        self.scene_ai = SceneAI(state=self.state)
        self.mixer_ai = MixerAI(
            state=self.state,
            emotion_ai=self.emotion_ai,
            scene_ai=self.scene_ai,
        )

        # Judge / Composer
        self.judge_ai = JudgeAI3(mode="normal")
        self.composer_ai = ComposerAI(self.llm_manager, refine_model="gpt52")

        # MemoryAI（保存先を default に統一：UIと一致させる）
        self.memory_ai = self._create_memory_ai(
            persona=persona,
            persona_id=self.runtime_persona_id,
            memory_model=memory_model,
        )

    # =========================================================
    # MemoryAI 新旧互換ファクトリ
    # =========================================================
    def _create_memory_ai(
        self,
        *,
        persona: Any,
        persona_id: str,
        memory_model: str,
    ) -> Any:
        persona_raw = getattr(persona, "raw", None)
        if not isinstance(persona_raw, dict):
            persona_raw = {}

        # v1.1+（keyword-only）を先に試す
        try:
            return MemoryAI(
                self.llm_manager,            # ★共有（分類器の二重生成回避）
                persona_id=persona_id,
                persona_raw=persona_raw,
                model_name=memory_model,
            )
        except TypeError:
            # 旧形式へ
            pass
        except Exception:
            pass

        # 旧形式（positional互換）
        try:
            return MemoryAI(
                self.llm_manager,
                persona_id=persona_id,
                model_name=memory_model,
            )
        except Exception as e:
            self.llm_meta.setdefault("memory_init_error", str(e))
            return None

    # =========================================================
    # 内部：AIManagerの enabled / priority を LLMManagerへ同期
    # =========================================================
    def _sync_ai_manager_settings(self) -> Dict[str, Any]:
        ai_state = st.session_state.get("ai_manager")
        if not isinstance(ai_state, dict):
            return {}

        enabled = ai_state.get("enabled_models")
        priority = ai_state.get("priority")

        if isinstance(enabled, dict):
            try:
                self.llm_manager.set_enabled_models(enabled)
            except Exception:
                pass

        out: Dict[str, Any] = {}
        if isinstance(priority, list):
            out["priority"] = [str(x) for x in priority if str(x).strip()]
        return out

    # =========================================================
    # 内部：Judgeに渡す候補だけ抽出（_meta/_system除外）
    # =========================================================
    @staticmethod
    def _extract_judge_candidates(models_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for name, info in (models_result or {}).items():
            if not isinstance(name, str) or name.startswith("_"):
                continue
            if not isinstance(info, dict):
                continue
            out[name] = {
                **info,
                "status": info.get("status", "unknown"),
                "text": info.get("text", "") or "",
            }
        return out

    # =========================================================
    # speak
    # =========================================================
    def speak(
        self,
        messages: List[Dict[str, str]],
        user_text: str = "",
        judge_mode: Optional[str] = None,
    ) -> str:
        if not messages:
            return ""

        # ★毎ターン同期（UIで殺した設定を確実に効かせる）
        sync = self._sync_ai_manager_settings()
        priority = sync.get("priority")

        # round_id を確定（この speak のターン番号）
        try:
            self.state["round_id"] = int(self.state.get("round_id", 0) or 0) + 1
        except Exception:
            self.state["round_id"] = 1
        round_id = int(self.state.get("round_id", 1) or 1)

        self.llm_meta["stage"] = "start"
        self.llm_meta["round_id"] = round_id

        try:
            InitAI.ensure_minimum(state=self.state, persona=self.persona)

            # -----------------------------------------
            # Memory context（可能なら）事前構築
            # -----------------------------------------
            self.llm_meta["stage"] = "memory_context"
            memory_context = ""
            try:
                if self.memory_ai is not None and hasattr(self.memory_ai, "build_memory_context"):
                    memory_context = str(
                        self.memory_ai.build_memory_context(
                            user_query=user_text or "",
                            max_items=5,
                        )
                        or ""
                    )
            except Exception as e:
                self.llm_meta["memory_context_error"] = str(e)
                memory_context = ""

            self.llm_meta["memory_context"] = memory_context

            # -----------------------------------------
            # Emotion override
            # -----------------------------------------
            self.llm_meta["stage"] = "mixer"
            emotion_override = self.mixer_ai.build_emotion_override()
            self.llm_meta["emotion_override"] = emotion_override

            # -----------------------------------------
            # Models collect
            # -----------------------------------------
            self.llm_meta["stage"] = "models_collect"
            results = self.models_ai.collect(
                messages,
                mode_current=judge_mode or "normal",
                emotion_override=emotion_override,
                reply_length_mode=self.llm_meta.get("reply_length_mode", "auto"),
            )
            self.llm_meta["models"] = results

            if not results:
                raise RuntimeError("ModelsAI2.collect returned empty dict")

            # -----------------------------------------
            # Judge（_meta/_systemを除外して渡す）
            # -----------------------------------------
            self.llm_meta["stage"] = "judge"
            judge_candidates = self._extract_judge_candidates(results)
            judge = self.judge_ai.run(
                judge_candidates,
                user_text=user_text,
                preferred_length_mode=str(st.session_state.get("reply_length_mode", "auto") or "auto"),
                priority=priority,
            )
            self.llm_meta["judge"] = judge

            # -----------------------------------------
            # Composer
            # -----------------------------------------
            self.llm_meta["stage"] = "composer"
            composed = self.composer_ai.compose(self.llm_meta)
            self.llm_meta["composer"] = composed

            final_text = (composed.get("text") or judge.get("chosen_text") or "").strip()

            # -----------------------------------------
            # Emotion analyze（失敗しても会話は継続）
            # -----------------------------------------
            self.llm_meta["stage"] = "emotion"
            try:
                emotion_res: EmotionResult = self.emotion_ai.analyze(
                    composer=composed,
                    memory_context=memory_context,
                    user_text=user_text,
                )
                self.llm_meta["emotion"] = emotion_res.to_dict()
                EmotionModel(result=emotion_res).sync_relationship_fields()
            except Exception as e:
                self.llm_meta["emotion_error"] = str(e)

            # -----------------------------------------
            # Memory update（必ず走らせて meta に残す）
            # -----------------------------------------
            self.llm_meta["stage"] = "memory_update"
            try:
                if self.memory_ai is None:
                    self.llm_meta["memory_update"] = {
                        "status": "skip",
                        "reason": "memory_ai_not_initialized",
                        "added": 0,
                    }
                else:
                    try:
                        mu = self.memory_ai.update_from_turn(
                            messages=messages,
                            final_reply=final_text,
                            round_id=round_id,
                        )
                    except TypeError:
                        mu = self.memory_ai.update_from_turn(
                            messages,
                            final_text,
                            round_id,
                        )

                    self.llm_meta["memory_update"] = mu if isinstance(mu, dict) else {"status": "ok", "raw": str(mu)}
            except Exception as e:
                self.llm_meta["memory_update_error"] = str(e)

            self.llm_meta["stage"] = "done"
            return final_text or "……"

        except Exception as e:
            self.llm_meta["stage"] = "fatal"
            err = {
                "error": str(e),
                "traceback": traceback.format_exc(limit=10),
                "round_id": round_id,
                "stage": self.llm_meta.get("stage"),
            }
            self.llm_meta.setdefault("errors", []).append(err)

            if LYRA_DEBUG:
                st.error("AnswerTalker.speak fatal error")
                st.exception(e)

            return "……（思考が途切れてしまったみたい）"
