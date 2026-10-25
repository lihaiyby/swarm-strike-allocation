"""最优解 oracle（P1.1）：给 a、b 提供**理论上界**，把绝对分变成最优性间隙。

为什么可以精确求解（而不是上 MILP）
------------------------------------
关键结构：我方无人机**同质**，且毁伤概率 ``p_j`` 只取决于敌机类型、与"派哪一架"无关。
因此 X（个体指派矩阵）对目标函数的影响**只通过各目标的架数** ``k_j = Σ_i x_ij`` 体现：

    a = Σ_j v_j · (1 - (1 - p_j)^{k_j})

于是"每架最多打一个目标"（C1）退化为一条单一的预算约束 ``Σ_j k_j ≤ |I|``，
C2（配额下限）退化为"每个目标的 k_j ∈ {0} ∪ [m_j, K_j]"。
整个问题就是一个**多选择背包**（multiple-choice knapsack），可用动态规划求**精确最优**。

这对任务 P1.1/P1.2 是决定性的简化：单场景精确最优只需 O(n·M·K)，毫秒级，
不依赖任何求解器的行为，也不会出现"求解器没跑好"的争议。

两个上界
--------
1. ``a_upper_bound``  —— **完美预见（上帝视角）**：在 t=0 就知道全部三波敌机，
   一次性对全体目标做背包。这是任何"只看当前态势"的算法都**达不到**的绝对天花板，
   但能给出"离理论极限还差多少"。
2. ``b_upper_bound``  —— 同样完美预见下，b 的理论最大：把"完成一个群"看作一个整体，
   群的代价 = Σ_{j∈k} m_j，收益 = V_k，又是一个 0/1 背包。

注意：二者都是**上界**，不是可达值。差距中既含"算法不够聪明"，也含"没有预见"。
把这两者拆开是 P1.2 的工作。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NEG = -1e18


@dataclass
class OracleResult:
    a_opt: float = 0.0
    b_opt: float = 0.0
    used: int = 0
    budget: int = 0
    counts: dict = field(default_factory=dict)      # enemy_id -> 架数
    b_used: float = 0.0                            # a 最优解所隐含的 b（真值团口径）
    b_groups: list = field(default_factory=list)   # b 最优时要完成的群（价值之和）

    @property
    def d_opt(self) -> float:
        return self.a_opt / max(self.used, 1)


def _quota(p: float, threshold: float) -> int:
    if p >= 1.0:
        return 1
    return int(np.ceil(np.log(1.0 - threshold) / np.log(1.0 - p)))


def _knapsack_a(values: np.ndarray, p_kill: np.ndarray, quotas: np.ndarray,
                budget: int, extra_cap: int = 20):
    """多选择背包：每个目标取 k=0 或 k∈[m_j, m_j+extra_cap]，预算 Σk ≤ budget。

    返回 (最优收益, counts 数组)。
    """
    n = len(values)
    M = int(budget)
    dp = np.full(M + 1, NEG)
    dp[0] = 0.0
    choice = np.zeros((n, M + 1), dtype=np.int16)

    for j in range(n):
        v, p, m = float(values[j]), float(p_kill[j]), int(quotas[j])
        kmax = min(M, m + extra_cap)
        ks = np.arange(m, kmax + 1)
        gains = v * (1.0 - (1.0 - p) ** ks)
        new = dp.copy()                     # 选项 k=0
        ch = np.zeros(M + 1, dtype=np.int16)
        for k, g in zip(ks, gains):
            if k > M:
                break
            cand = dp[: M + 1 - k] + g      # 向量化：整体右移 k 位再取 max
            sl = new[k:]
            better = cand > sl
            sl[better] = cand[better]
            ch[k:][better] = k
        dp, choice[j] = new, ch

    best_b = int(np.argmax(dp))
    counts = np.zeros(n, dtype=int)
    b = best_b
    for j in range(n - 1, -1, -1):
        k = int(choice[j, b])
        counts[j] = k
        b -= k
    return float(dp[best_b]), counts


def _knapsack_b(group_values: np.ndarray, group_costs: np.ndarray, budget: int):
    """0/1 背包：选若干群"整群完成"，代价 Σm_j，收益 V_k，预算 Σ ≤ budget。"""
    M = int(budget)
    dp = np.full(M + 1, NEG)
    dp[0] = 0.0
    take = np.zeros((len(group_values), M + 1), dtype=bool)
    for i, (val, cost) in enumerate(zip(group_values, group_costs)):
        cost = int(cost)
        if cost <= 0 or cost > M:
            continue
        new = dp.copy()
        tk = np.zeros(M + 1, dtype=bool)
        for b in range(M - cost + 1):
            if dp[b] <= NEG / 2:
                continue
            cand = dp[b] + val
            if cand > new[b + cost]:
                new[b + cost] = cand
                tk[b + cost] = True
        dp, take[i] = new, tk
    best_b = int(np.argmax(dp))
    chosen = []
    b = best_b
    for i in range(len(group_values) - 1, -1, -1):
        if take[i, b]:
            chosen.append(i)
            b -= int(group_costs[i])
    return float(dp[best_b]), chosen


def scene_upper_bound(scene, model) -> OracleResult:
    """单场景的完美预见上界（a 与 b 各自最优）。

    - a_opt：多选择背包，预算 = 全场编制 |I| = Σ 我方无人机数
    - b_opt：0/1 背包，把"整群完成"当整体
    - b_used：a 最优解所隐含的 b（用于检查"最大化 a 会牺牲多少 b"）
    """
    units = [u for w in scene.waves for u in w.units]
    eids = [u.id for u in units]
    values = np.array([u.value for u in units], dtype=float)
    p_kill = np.array([u.p_kill for u in units], dtype=float)
    quotas = np.array([_quota(p, model.kill_threshold) for p in p_kill], dtype=int)
    budget = len(scene.friendly)

    a_opt, counts = _knapsack_a(values, p_kill, quotas, budget)

    # a 最优解隐含的 b（真值团口径）
    idx = {e: i for i, e in enumerate(eids)}
    P = {e: float(1.0 - (1.0 - p_kill[i]) ** counts[i]) for i, e in enumerate(eids)}
    groups = scene.ground_truth_groups()
    from .b_metric import group_elimination_rate
    vals = {e: float(v) for e, v in zip(eids, values)}
    b_used, _, _ = group_elimination_rate(P, groups, vals, model.kill_threshold)

    # b 的上界：每个真值团整群完成的代价 = Σ m_j
    gvals, gcosts = [], []
    for members in groups:
        gvals.append(sum(vals[e] for e in members))
        gcosts.append(sum(int(quotas[idx[e]]) for e in members))
    b_total = sum(gvals)
    b_num_opt, chosen = _knapsack_b(np.array(gvals), np.array(gcosts), budget)
    b_opt = b_num_opt / b_total if b_total > 0 else 0.0

    return OracleResult(
        a_opt=a_opt, b_opt=b_opt, used=int(counts.sum()), budget=int(budget),
        counts={e: int(counts[i]) for i, e in enumerate(eids)},
        b_used=b_used, b_groups=chosen)
