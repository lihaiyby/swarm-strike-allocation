"""随机可行解（下界基线）——评估参照策略，**不是候选算法**。

用途（任务 P1.3）
-----------------
给每条指标定一个"地板"：任何算法都不应该比它更差。
有了 `scene_upper_bound`（天花板）与本模块（地板），任一算法的分数都可以被读成
"在地板与天花板之间的位置"，而不是一个孤立的绝对数。

策略
----
把目标随机打乱，依次整包（k = 配额 m_j）填充，直到剩余兵力不足或目标取完。
这样**必然满足** C1（单架单目标）与 C2（打则必达配额），且不做任何价值/威胁排序，
因此是"完全不思考"的下界。

可复现：随机种子由 `(state.t, 最小敌机 id)` 派生，不依赖全局随机状态。
"""
from __future__ import annotations

import numpy as np

from ..model import Model
from .base import AlgoBase, register


@register
class RandomFeasible(AlgoBase):
    name = "random_feasible"

    def solve(self, state, model: Model) -> np.ndarray:
        eids, fids = state.enemy_ids, state.pool_ids
        X = np.zeros((len(fids), len(eids)), dtype=int)
        if not eids or not fids:
            return X

        seed = (int(round(state.t * 1000)) * 100003 + int(eids[0])) % (2 ** 32)
        rng = np.random.default_rng(seed)

        values, p_kill, epos = state.enemy_matrix()
        quotas = np.array([model.quota(p, model.kill_threshold) for p in p_kill])

        order = rng.permutation(len(eids))
        fpos = state.pool_matrix()
        avail = np.ones(len(fids), dtype=bool)
        remaining = int(avail.sum())

        for j in order:
            k = int(quotas[j])
            if k <= 0 or remaining < k:      # 兵力不足则整包放弃（C2）
                continue
            d = np.linalg.norm(fpos - epos[j], axis=1)
            d[~avail] = np.inf
            idx = np.argsort(d)[:k]          # 只决定"哪几架"，不决定"打哪个"
            X[idx, j] = 1
            avail[idx] = False
            remaining -= k
        return X
