from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal


TimeSlot = Literal["morning", "lunch", "after_school", "night"]


@dataclass
class WorldContext:
    """
    world_state の“運用しやすい窓口”となる構造体。

    - 保存先（唯一の正）は state["world_state"] として維持しつつ、
      普段の読み書きはこの構造体でやり取りする。
    - InitAI が WorldContext <-> world_state を変換/同期する。
    """

    # a) プレイヤー名
    player_name: str = "プレイヤー"

    # b) 相手キャラ（Persona 由来をデフォルトにし、必要なら上書き）
    partner_role: str = "floria"
    partner_name: str = "フローリア"

    # c) プレイヤー所在地 / d) 相手所在地
    # None の場合は InitAI が規定値を補完する（例: "{player_name}の部屋"）
    player_location: Optional[str] = None
    partner_location: Optional[str] = None

    # e) 時刻
    time_slot: TimeSlot = "morning"
    time_str: str = "07:30"

    # f) モブの有無
    others_present: bool = False

    # 追加（当面の既定値）
    weather: str = "clear"
