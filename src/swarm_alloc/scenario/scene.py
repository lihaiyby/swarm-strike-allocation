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

    def ground_truth_groups(self) -> list:
        """真值团：按生成时的 (波次, cluster) 划分，噪点各自独立成团。

        与感知层的动态 DBSCAN 分组无关——这是**战场客观事实**，
        用作评估指标 b 的群定义（见 docs/擂台改造-任务清单.md D2 = ②）。

        返回 list[list[enemy_id]]：
        - 覆盖本场景**全部**敌机，且每个敌机恰好出现一次（划分，不是覆盖）；
        - 顺序确定（波次升序 → 团编号升序 → 噪点按 id 升序），保证可复现。
        """
        groups = []
        for wave in self.waves:
            by_cluster = {}
            noise = []
            for u in wave.units:
                if u.cluster >= 0:
                    by_cluster.setdefault(u.cluster, []).append(u.id)
                else:
                    noise.append([u.id])
            for c in sorted(by_cluster):
                groups.append(sorted(by_cluster[c]))
            groups.extend(sorted(noise))
        return groups
