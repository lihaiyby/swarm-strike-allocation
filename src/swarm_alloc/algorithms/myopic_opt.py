"""近视最优（myopic optimum）——评估用参照策略，**不是候选算法**。

用途（任务 P1.6）
-----------------
完美预见上界 `evaluation.oracle.scene_upper_bound` 假定 t=0 就知道全部三波敌机，
任何"只看当前态势"的算法都达不到。为把间隙拆成两部分：

    a_opt(完美预见) − a_算法  =  预见价值  +  能力差距

需要一个"只看当前态势、但每次决策都做到精确最优"的参照策略，即本模块。

策略
----
每个触发点，对**当前在场敌机 + 当前剩余兵力**求解精确的多选择背包
（每目标 k=0 或 k≥配额，预算 Σk ≤ 剩余兵力），得到各目标架数后按最近邻落到具体无人机。

关键点：它**不知道后面还有几波**，因此会把剩余兵力一次性用完——
这正好量化了"缺乏预见"的代价。如果它显著劣于规则式的 greedy，
就说明**这套想定里真正的本领是跨波次的兵力克制，而不是单轮优化**。
"""
from __future__ import annotations

import numpy as np

from ..evaluation.oracle import _knapsack_a
from ..model import Model
from .base import AlgoBase, register


@register
class MyopicOpt(AlgoBase):
    name = "myopic_opt"

    EXTRA_CAP = 20          # 每目标在配额之上最多再给多少架

    def solve(self, state, model: Model) -> np.ndarray:
        eids, fids = state.enemy_ids, state.pool_ids
        X = np.zeros((len(fids), len(eids)), dtype=int)
        if not eids or not fids:
            return X

        values, p_kill, epos = state.enemy_matrix()
        quotas = np.array([model.quota(p, model.kill_threshold) for p in p_kill])
        budget = len(fids)

        # 只读 state，调框架的配额；背包本身是纯优化工具，不含毁伤/约束公式
        _, counts = _knapsack_a(values, p_kill, quotas, budget, self.EXTRA_CAP)

        fpos = state.pool_matrix()
        avail = np.ones(len(fids), dtype=bool)
        for j in np.argsort(-values):
            k = int(counts[j])
            if k <= 0:
                continue
            if int(avail.sum()) < k:       # 兵力不足则整包放弃（C2：打则必达配额）
                continue
            d = np.linalg.norm(fpos - epos[j], axis=1)
            d[~avail] = np.inf
            idx = np.argsort(d)[:k]
            X[idx, j] = 1
            avail[idx] = False
        return X
