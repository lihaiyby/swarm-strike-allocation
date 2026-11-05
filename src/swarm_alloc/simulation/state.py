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
    # ---- 【批注 B3】我方分组与配对（仅在 grouping 启用时非空）
    friendly_groups: dict = field(default_factory=dict)   # fid -> 我方组号
    group_match: dict = field(default_factory=dict)       # 我方组号 -> 敌群 id

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

    # ---- D6 ----
    def pair_p_matrix(self, model) -> np.ndarray:
        """(我机 × 目标) 的实际单架次毁伤概率矩阵 ``p_ij``（相对距离口径）。

        关闭距离衰减时退化为"每目标一个常数"的广播结果，与旧口径一致。
        算法可用它来区分"近机"与"远机"——这是 D6 引入的唯一新信息。
        """
        _, p, epos = self.enemy_matrix()
        fpos = self.pool_matrix()
        d = np.linalg.norm(fpos[:, None, :] - epos[None, :, :], axis=2)
        return model.pair_p_matrix(p, d)

    # ---- 【批注 B3】C3 允许矩阵 ----
    def c3_allowed(self) -> np.ndarray:
        """(我机 × 目标) 的布尔矩阵：True 表示该机允许打该目标。

        未启用分组时全为 True（等价于无 C3）。
        """
        n_f, n_e = len(self.pool_ids), len(self.enemy_ids)
        if not self.friendly_groups or not self.group_match:
            return np.ones((n_f, n_e), dtype=bool)
        enemy_group_of = {u: g.id for g in self.groups for u in g.member_ids}
        out = np.zeros((n_f, n_e), dtype=bool)
        for i, fid in enumerate(self.pool_ids):
            tgt = self.group_match.get(self.friendly_groups.get(fid))
            if tgt is None:
                continue
            for j, eid in enumerate(self.enemy_ids):
                if enemy_group_of.get(eid) == tgt:
                    out[i, j] = True
        return out

