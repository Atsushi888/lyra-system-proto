# actors/scene/world_context.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorldContext:
    """
    ゲーム世界の現在状態を表す正規データ構造。
    world_state / manual_controls / narrator / scene の
    すべての初期化元になる。
    """

    # ===== キャラクター =====
    player_name: str
    partner_name: str

    # ===== 位置 =====
    player_location: str
    partner_location: str

    # ===== 時間 =====
    time_slot: str = "morning"   # morning / lunch / after_school / night
    time_str: str = "07:30"

    # ===== 環境 =====
    others_present: bool = False

    # ===== 補助 =====
    weather: str = "clear"

    @property
    def party_mode(self) -> str:
        if self.player_location == self.partner_location:
            return "both"
        return "alone"
