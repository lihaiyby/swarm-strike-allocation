"""边际收益贪心 baseline（理论标尺，毫秒级）。

流程（方案04-v3 §3）：
1. 群级配对：敌群按威胁度降序，可行性恢复（配额需求超资源池则放弃低威胁群）；
   编组 = 按配额需求从资源池取"离该群质心最近"的无人机；
2. 个体分配：目标 j 展开 m_j 个打击槽位，群内按最近距离指派。
"""
from __future__ import annotations

import numpy as np

from ..model import Model
from .base import AlgoBase, register


@register
class Greedy(AlgoBase):
    name = "greedy"

    def solve(self, state, model: Model) -> np.ndarray:
        eids, fids = state.enemy_ids, state.pool_ids
        X = np.zeros((len(fids), len(eids)), dtype=int)
        if not eids or not fids:
            return X
        eidx = {e: j for j, e in enumerate(eids)}

        values, p_kill, epos = state.enemy_matrix()
        quotas = np.array([model.quota(p, model.kill_threshold) for p in p_kill])
        group_of_enemy = {uid: g.id for g in state.groups for uid in g.member_ids}

        # 各群配额需求（只统计仍在空的敌机）
        demand = {}
        for g in state.groups:
            js = [eidx[uid] for uid in g.member_ids if uid in eidx]
            demand[g.id] = int(quotas[js].sum()) if js else 0

        kept = model.select_groups(state.groups, demand, len(fids))

        # 编组划分：按威胁度顺序，依次取离群质心最近的 demand_k 架
        fpos = state.pool_matrix()
        available = np.ones(len(fids), dtype=bool)
        for g in kept:
            js = [eidx[uid] for uid in g.member_ids if uid in eidx]
            if not js:
                continue
            centroid = epos[js].mean(axis=0)
            d = np.linalg.norm(fpos - centroid, axis=1)
            d[~available] = np.inf
            take = min(demand[g.id], int(available.sum()))
            members = np.argsort(d)[:take]
            available[members] = False

            # 群内槽位展开：目标 j 需要 quotas[j] 架，指派最近的编组内无人机
            slots = [j for j in js for _ in range(int(quotas[j]))]
            for s, j in enumerate(slots[: len(members)]):
                dists = np.linalg.norm(fpos[members] - epos[j], axis=1)
                li = int(np.argmin(dists))
                X[members[li], j] = 1
                members = np.delete(members, li)
        return X
