"""毕义军 PC_ISA：贪心预分配 + 改进模拟退火（注册名 `pc_isa`）。

文献
----
毕义军, 王燮, 蒋田. 基于联盟博弈与模拟退火的无人机集群任务分配研究[J].
现代电子技术, 2026, 49(9).

原文框架（两阶段）
------------------
- **联盟博弈**：某一组无人机协同完成一个任务，构成一个"联盟"；目标是最小化**联盟代价函数**。
- **阶段 1 — 局部贪心动态算法**：快速给出初始分配，**缩减不可行域**；
- **阶段 2 — 改进模拟退火**：引入**自适应温度衰减**，最小化联盟代价；
- **硬约束**：任务优先级。

移植到本项目的适配
------------------
| 原文 | 本项目 | 处理 |
|---|---|---|
| 联盟 = 多机完成同一任务 | 同 | 一个联盟对应一个敌群的任务集合 |
| 联盟代价函数 | 本项目用"负收益 + 距离代价" | $C=-\\sum_j v_jP_j+\\lambda\\sum d_{ij}x_{ij}/R$ |
| 任务优先级硬约束 | 配额 C2（0 或 ≥ $m_j$） | 用**惩罚项**处理：未满配额的目标罚 $B\\cdot m_j$ |
| 自适应温度衰减 | 同 | $T_{k+1}=\\alpha T_k$，并在停滞时回温 |

算法结构
--------
1. **阶段 1**：贪心预分配（按边际收益 − 距离代价选「目标×机」对），得到可行初始解；
2. **阶段 2**：模拟退火。邻域动作 = 把一架已分配的机改派到另一个还有余量的目标；
   接受准则 Metropolis；温度按 $\\alpha$ 衰减，连续若干步无改进则回温（自适应）；
3. **整包裁决**：收尾时保证 C2。

【批注 B3】若启用组对组，通过 `state.c3_allowed()` 限制可选对。
"""
from __future__ import annotations

import numpy as np

from ..model import Model
from .base import AlgoBase, register


@register
class PCISA(AlgoBase):
    name = "pc_isa"

    LAMBDA_COST: float = 0.35     # 联盟代价中的距离权重
    T0: float = 2.0               # 初始温度
    T_MIN: float = 1e-3
    ALPHA: float = 0.90           # 降温系数
    ITERS: int = 600              # 退火步数
    NO_IMPROVE: int = 80          # 连续无改进则回温
    PENALTY: float = 50.0         # 未满配额惩罚系数

    def solve(self, state, model: Model) -> np.ndarray:
        eids, fids = state.enemy_ids, state.pool_ids
        X = np.zeros((len(fids), len(eids)), dtype=int)
        if not eids or not fids:
            return X

        values, p_kill, epos = state.enemy_matrix()
        fpos = state.pool_matrix()
        quotas = np.array([model.quota(p, model.kill_threshold) for p in p_kill])
        eidx = {e: j for j, e in enumerate(eids)}

        gmembers = {g.id: [eidx[u] for u in g.member_ids if u in eidx]
                    for g in state.groups}
        demand = {g: int(quotas[ms].sum()) for g, ms in gmembers.items() if ms}
        gvalue = {g: float(values[ms].sum()) for g, ms in gmembers.items() if ms}
        kept = model.select_groups(state.groups, demand, len(fids), gvalue)
        allowed = {u for g in kept for u in g.member_ids}
        cap = np.array([int(quotas[j]) if eids[j] in allowed else 0
                        for j in range(len(eids))], dtype=int)
        if cap.sum() == 0:
            return X

        dist = np.linalg.norm(fpos[:, None, :] - epos[None, :, :], axis=2)
        dn = dist / (np.linalg.norm(epos, axis=1).max() + 1e-9)
        c3 = state.c3_allowed()

        # ---------- 阶段 1：贪心预分配（缩减不可行域） ----------
        count = np.zeros(len(eids), dtype=int)
        for _ in range(int(cap.sum())):
            used = X.any(axis=1)
            cand = (~used)[:, None] & (count[None, :] < cap[None, :]) & c3
            if not cand.any():
                break
            gain = values * ((1 - (1 - p_kill) ** (count + 1))
                             - (1 - (1 - p_kill) ** count))
            gain = np.where(count < cap, gain, -np.inf)
            dm = np.where(cand, dn, np.inf)
            bi = np.argmin(dm, axis=0)
            bd = dm[bi, np.arange(len(eids))]
            ok = np.isfinite(bd)
            if not ok.any():
                break
            u = np.where(ok, gain - self.LAMBDA_COST * bd, -np.inf)
            j = int(np.argmax(u))
            if not np.isfinite(u[j]) or u[j] <= 0:
                break
            X[int(bi[j]), j] = 1
            count[j] += 1

        # ---------- 阶段 2：改进模拟退火 ----------
        X = self._anneal(X, count, cap, values, p_kill, dn, c3)

        # ---------- 整包裁决（C2） ----------
        X = self._repair(X, cap, dist, c3)
        return X

    # ------------------------------------------------------------------
    def _cost(self, X, count, cap, values, p_kill, dn):
        """联盟代价 = 负收益 + 距离代价 + 未满配额惩罚。"""
        P = 1.0 - (1.0 - p_kill) ** count
        gain = float(np.dot(values, P))
        distc = float((dn * X).sum())
        partial = ((count > 0) & (count < cap))
        pen = self.PENALTY * float(cap[partial].sum())
        return -gain + self.LAMBDA_COST * distc + pen

    def _anneal(self, X, count, cap, values, p_kill, dn, c3):
        n_f, n_e = X.shape
        rng = np.random.default_rng(int(values.sum() * 1000) % (2 ** 32))
        cur = self._cost(X, count, cap, values, p_kill, dn)
        best_X, best_cost = X.copy(), cur
        T = self.T0
        stall = 0

        ii, jj = np.nonzero(X)
        assigned = list(zip(ii.tolist(), jj.tolist()))
        if not assigned:
            return X

        for _ in range(self.ITERS):
            if T <= self.T_MIN:
                break
            # 邻域：随机挑一架已分配的机，改派到另一个有余量且允许的目标
            i, j0 = assigned[rng.integers(len(assigned))]
            cand = np.nonzero((count < cap) & c3[i] & (np.arange(n_e) != j0))[0]
            if len(cand) == 0:
                continue
            j1 = int(cand[rng.integers(len(cand))])
            # 试走
            X[i, j0] = 0
            count[j0] -= 1
            X[i, j1] = 1
            count[j1] += 1
            new = self._cost(X, count, cap, values, p_kill, dn)
            d = new - cur
            if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
                cur = new
                assigned[assigned.index((i, j0))] = (i, j1)
                if cur < best_cost - 1e-9:
                    best_cost, best_X = cur, X.copy()
                    stall = 0
                else:
                    stall += 1
            else:                                   # 回滚
                X[i, j1] = 0
                count[j1] -= 1
                X[i, j0] = 1
                count[j0] += 1
            T *= self.ALPHA
            if stall >= self.NO_IMPROVE:            # 自适应回温
                T = min(self.T0, T * 4.0)
                stall = 0
        return best_X

    # ------------------------------------------------------------------
    def _repair(self, X, cap, dist, c3):
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
                for i in sorted(cand, key=lambda i: float(dist[i, j]))[:need]:
                    X[i, j] = 1
                    idle.remove(i)
            else:
                for i in np.nonzero(X[:, j])[0]:
                    X[i, j] = 0
                    idle.append(int(i))
        return X
