"""场景数据结构定义。"""

from dataclasses import dataclass

import numpy as np


@dataclass
class EnemyUnit:
    id: int
    pos: np.ndarray      # (3,)，|pos| = R（外球面生成）
    vel: np.ndarray      # (3,)，直线匀速指向球心
    speed: float         # m/s
    type_name: str       # recon / suicide_strike / fighter / area_damage
    value: int           # 10 / 5 / 2
    maneuver: int        # 机动档 10 / 5 / 2
    p_kill: float        # 我方单机对其毁伤概率
    is_noise: bool       # 是否为散点（噪点，独立成群）
    cluster: int         # 生成时的团编号（-1 = 噪点）


@dataclass
class Wave:
    t: float             # 出现时刻（s）
    units: list


@dataclass
class FriendlyUnit:
    id: int
    pos: np.ndarray      # (3,)，内球面（防线 700m）
    speed: float


@dataclass
class Scene:
    seed: int
    R: float             # 战场半径 m
    T: float             # 战局总时长 s
    waves: list          # list[Wave]
    friendly: list       # list[FriendlyUnit]

    @property
    def n_enemy(self) -> int:
        return sum(len(w.units) for w in self.waves)
