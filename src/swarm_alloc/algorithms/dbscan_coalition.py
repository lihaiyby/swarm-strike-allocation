r"""常绪成 改进 DBSCAN + 贪心联盟形成博弈（注册名 `dbscan_coalition`）。

文献
----
常绪成, 党帅龙, 朱锋, 等. 非均衡场景下多无人机混合任务分配算法[J/OL].
航空学报, 2026. doi:10.7527/S1000-6893.2026.33601

原文框架（集中-分布式混合）
--------------------------
1. **改进 DBSCAN**：k-距离图拐点识别实现邻域参数自适应；**近簇匹配的噪声点分层处理**，
   生成空间紧凑的任务簇以降低问题规模；
2. **贪心联盟形成博弈**：基于任务簇构建联盟；引入**资源匹配与续航约束的初始化机制**、
   **无人机多策略决策**，以**效用函数**为量化依据、**功利主义顺序**为决策准则，
   引导系统收敛至**纳什均衡**；
3. 效用函数建模了**资源、路径长度与能耗**三个目标。

移植到本项目的适配
------------------
| 原文 | 本项目 | 处理 |
|---|---|---|
| 算法内部做 DBSCAN 分组 | 平台感知层已统一做 DBSCAN + TOPSIS | **不重复实现**：直接使用 `state.groups`（保证与其他算法同一起点） |
| 续航/能耗约束 | 无航程建模 | 用**距离代价**近似：$\\lambda_c\cdot d_{ij}/R$ |
| 路径长度目标 | 同上 | 同上 |
| 联盟 = 多机协同攻击一个任务簇 | 同 | 一个联盟对应一个敌群 |
| 纳什均衡 | 单机改换联盟直到无人愿意单独改变选择 | 实现为"单机改换改进轮" |

算法结构
--------
1. **初始化（资源匹配）**：按各群配额需求比例，给出初始联盟规模；
2. **贪心联盟形成（功利主义顺序）**：
   每一步在所有"目标 × 可用机"中选**边际效用最大**的一对
   $$u_{ij}=\\underbrace{v_j\\big(P_j(k_j+1)-P_j(k_j)\\big)}_{\\text{边际毁伤收益}}
            -\\lambda_c\\frac{d_{ij}}{R}$$
3. **纳什改进**：逐机检查"改换目标是否提高自身效用"，直到无人愿意改变（或达迭代上限）；
4. **整包裁决**：保证 C2。

【批注 B3】若启用组对组，则通过 `state.c3_allowed()` 限制可选（机, 目标）对。
"""
from __future__ import annotations

import numpy as np

from ..model import Model
from .base import AlgoBase, register


@register
class DBSCANCoalition(AlgoBase):
    name = "dbscan_coalition"

    # 效用函数中的路径/能耗代价权重（原文 λ 的移植）
    LAMBDA_COST: float = 0.35
    # 纳什改进的最大轮数
    MAX_NASH_ITERS: int = 8

    def solve(self, state, model: Model) -> np.ndarray:
        eids, fids = state.enemy_ids, state.pool_ids
        X = np.zeros((len(fids), len(eids)), dtype=int)
        if not eids or not fids:
            return X

        values, p_kill, epos = state.enemy_matrix()
        fpos = state.pool_matrix()
        quotas = np.array([model.quota(p, model.kill_threshold) for p in p_kill])
        eidx = {e: j for j, e in enumerate(eids)}

        # ---------- 任务簇（原文的任务簇 = 本项目的敌群） ----------
        gmembers = {g.id: [eidx[u] for u in g.member_ids if u in eidx]
                    for g in state.groups}
        demand = {g: int(quotas[ms].sum()) for g, ms in gmembers.items() if ms}
        gvalue = {g: float(values[ms].sum()) for g, ms in gmembers.items() if ms}

        # 可行性恢复（与其他算法同口径）
        kept = model.select_groups(state.groups, demand, len(fids), gvalue)
        allowed_targets = {u for g in kept for u in g.member_ids}
        cap = np.array([int(quotas[j]) if eids[j] in allowed_targets else 0
                        for j in range(len(eids))], dtype=int)
        if cap.sum() == 0:
            return X

        dist = np.linalg.norm(fpos[:, None, :] - epos[None, :, :], axis=2)
        dn = dist / (np.linalg.norm(epos, axis=1).max() + 1e-9)   # 归一化距离
        c3 = state.c3_allowed()                                    # 【批注 B3】

        count = np.zeros(len(eids), dtype=int)

        # ---------- 阶段 1+2：贪心联盟形成（功利主义顺序） ----------
        for _ in range(int(cap.sum())):
            used = X.any(axis=1)
            cand = (~used)[:, None] & (count[None, :] < cap[None, :]) & c3
            if not cand.any():
                break
            # 每个目标的边际毁伤收益
            gain = values * ((1.0 - (1.0 - p_kill) ** (count + 1))
                             - (1.0 - (1.0 - p_kill) ** count))
            gain = np.where(count < cap, gain, -np.inf)
            # 每个目标可用的最近机
            d_masked = np.where(cand, dn, np.inf)
            best_i = np.argmin(d_masked, axis=0)
            best_d = d_masked[best_i, np.arange(len(eids))]
            ok = np.isfinite(best_d)
            if not ok.any():
                break
            u = gain - self.LAMBDA_COST * best_d
            u = np.where(ok, u, -np.inf)
            j = int(np.argmax(u))
            if not np.isfinite(u[j]) or u[j] <= 0:
                break
            i = int(best_i[j])
            X[i, j] = 1
            count[j] += 1

        # ---------- 阶段 3：纳什改进（单机改换联盟） ----------
        X = self._nash_improve(X, count, cap, values, p_kill, dn, c3)

        # ---------- 阶段 4：整包裁决（C2） ----------
        X = self._repair(X, cap, dist, c3)
        return X

    # ------------------------------------------------------------------
    def _own_utility(self, i, j, count, values, p_kill, dn):
        """单架机对目标 j 的边际效用（含距离代价）。"""
        g = values[j] * ((1 - (1 - p_kill[j]) ** (count[j] + 1))
                         - (1 - (1 - p_kill[j]) ** count[j]))
        return float(g - self.LAMBDA_COST * dn[i, j])

    def _nash_improve(self, X, count, cap, values, p_kill, dn, c3):
        """逐机尝试改换目标，若自身效用提高则移动；直到无人愿意改变。

        "自身效用"按该机加入后的**边际效用**计（含距离代价），
        与原文"以效用函数为量化依据引导收敛至纳什均衡"对应。
        """
        n_f, n_e = X.shape
        for _ in range(self.MAX_NASH_ITERS):
            moved = False
            for i in range(n_f):
                cur = np.nonzero(X[i])[0]
                cur_j = int(cur[0]) if len(cur) else -1
                cur_u = -np.inf
                if cur_j >= 0:
                    c = count.copy()
                    c[cur_j] -= 1
                    cur_u = self._own_utility(i, cur_j, c, values, p_kill, dn)
                best_j, best_u = cur_j, cur_u
                for j in range(n_e):
                    if j == cur_j or not c3[i, j] or count[j] >= cap[j]:
                        continue
                    u = self._own_utility(i, j, count, values, p_kill, dn)
                    if u > best_u:
                        best_j, best_u = j, u
                if best_j != cur_j:
                    if cur_j >= 0:
                        X[i, cur_j] = 0
                        count[cur_j] -= 1
                    X[i, best_j] = 1
                    count[best_j] += 1
                    moved = True
            if not moved:
                break
        return X

    def _repair(self, X, cap, dist, c3):
        """整包裁决：未满配额的目标要么补齐、要么整批撤单。"""
        count = X.sum(axis=0)
        partial = [j for j in range(len(cap)) if 0 < count[j] < cap[j]]
        if not partial:
            return X
        busy = set(np.nonzero(X.any(axis=1))[0].tolist())
        idle = [i for i in range(X.shape[0]) if i not in busy]
        for j in sorted(partial):
            need = int(cap[j] - count[j])
            cand = [i for i in idle if c3[i, j]]
            if len(cand) >= need:
                pick = sorted(cand, key=lambda i: float(dist[i, j]))[:need]
                for i in pick:
                    X[i, j] = 1
                    idle.remove(i)
            else:
                for i in np.nonzero(X[:, j])[0]:
                    X[i, j] = 0
                    idle.append(int(i))
        return X
