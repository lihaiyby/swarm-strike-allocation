"""态势数据结构：算法接口的统一输入。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class State:
    t: float
    enemies: dict            # enemy_id -> EnemyUnit（当前在空，pos 已推进到时刻 t）
    groups: list             # list[Group]，threat 已由 TOPSIS 填好
    pool: dict               # friendly_id -> FriendlyUnit（可用）
    frozen_ids: list = field(default_factory=list)   # 已下达指令，不可改派

    @property
    def enemy_ids(self) -> list:
        return sorted(self.enemies.keys())

    @property
    def pool_ids(self) -> list:
        return sorted(self.pool.keys())

    def enemy_matrix(self) -> tuple:
        """返回 (values, p_kill, positions)，按 enemy_ids 顺序。"""
        us = [self.enemies[i] for i in self.enemy_ids]
        return (np.array([u.value for u in us], dtype=float),
                np.array([u.p_kill for u in us], dtype=float),
                np.vstack([u.pos for u in us]))

    def pool_matrix(self) -> np.ndarray:
        return np.vstack([self.pool[i].pos for i in self.pool_ids])
