"""孙鹏耀 改进双种群遗传算法（注册名 `ga_dual`）。

文献
----
孙鹏耀, 黄炎焱, 居界林. 基于改进双种群遗传算法的无人机集群任务分配方法[J/OL].
指挥控制与仿真, 2026. https://link.cnki.net/urlid/32.1759.tj.20260119.1501.008

原文
----
1. 分析"察打评"（侦察—打击—评估）作战过程，构建**宽约束**任务分配模型；
2. 提出**改进双种群遗传算法**：**主种群加入变邻域搜索（VNS）**以提高求解效率；
   双种群 = 两个独立演化、参数不同的种群（保持多样性）。

移植到本项目的适配
------------------
| 原文 | 本项目 |
|---|---|
| 察打评多类任务 | 只有"打击"一类（侦察/评估不建模）|
| 宽约束模型 | 用配额 C2 + C1 代替（更严） |
| 双种群 | **主种群**（大种群、强交叉）+ **辅助种群**（小种群、高变异），交替演化并共享精英 |
| 主种群 VNS | 对每代最优个体做变邻域局部搜索：把一个单位的配额从目标 A 移到目标 B |
| 基因编码 | 与 `ga_iap` 一致：每目标分配架数向量（我方同质，个体指派由解码按距离决定） |

与 `ga_iap` 的区别：单目标（只最大化收益）、双种群、带 VNS；`ga_iap` 是双目标 NSGA-III。
"""
from __future__ import annotations

import numpy as np

from ..model import Model
from ._common import marginal, prepare, repair_c2
from .base import AlgoBase, register


@register
class GADual(AlgoBase):
    name = "ga_dual"

    POP_MAIN: int = 40
    POP_AUX: int = 16
    GENS: int = 60
    PC_MAIN: float = 0.85
    PM_MAIN: float = 0.08
    PC_AUX: float = 0.60
    PM_AUX: float = 0.20
    VNS_ITERS: int = 20
    LAMBDA_COST: float = 0.20

    def solve(self, state, model: Model) -> np.ndarray:
        ctx = prepare(state, model)
        X = np.zeros((len(state.pool_ids), len(state.enemy_ids)), dtype=int)
        if ctx is None:
            return X

        n = len(ctx.eids)
        cap = ctx.cap
        rng = np.random.default_rng(int(ctx.values.sum() * 1000 + state.t * 7) % (2 ** 32))

        def fitness(pop):
            """收益 − 距离代价（用"每目标取最近 k 架"的距离近似）。"""
            P = 1.0 - (1.0 - ctx.p_kill) ** pop
            gain = (ctx.values[None, :] * P).sum(axis=1)
            order = np.argsort(ctx.dn, axis=0)            # (n_f, n_e)
            cost = np.zeros(len(pop))
            for k in range(len(pop)):
                s = 0.0
                for j in np.nonzero(pop[k])[0]:
                    m = int(pop[k, j])
                    s += float(ctx.dn[order[:m, j], j].sum())
                cost[k] = s
            return gain - self.LAMBDA_COST * cost

        def init(pop_size):
            p = np.zeros((pop_size, n), dtype=int)
            for k in range(pop_size):
                budget = len(ctx.fids)
                for j in rng.permutation(n):
                    c = int(cap[j])
                    if c and c <= budget and rng.random() < 0.6:
                        p[k, j] = c
                        budget -= c
            return p

        def selection(pop, fit):
            idx = rng.choice(len(pop), size=2, replace=False)
            return pop[idx[np.argmax(fit[idx])]]

        def crossover(a, b):
            if rng.random() > self.PC_MAIN:
                return a.copy()
            m = rng.random(n) < 0.5
            return np.where(m, a, b)

        def mutate(x, pm):
            m = rng.random(n) < pm
            x = np.where(m, rng.integers(0, 2, size=n) * cap, x)
            budget = len(ctx.fids)
            while x.sum() > budget:
                j = rng.integers(n)
                x[j] = 0
            return x

        def vns(x):
            """变邻域搜索：尝试把一个单位的配额从目标 A 移到目标 B。"""
            best, best_f = x.copy(), fitness(x[None, :])[0]
            for _ in range(self.VNS_ITERS):
                a, b = rng.integers(n, size=2)
                if a == b:
                    continue
                y = best.copy()
                if y[a] >= cap[a] and cap[a] > 0:
                    y[a] = 0
                else:
                    y[a] = min(cap[a], y[a] + 1)
                if y[b] > 0:
                    y[b] = 0
                else:
                    y[b] = min(cap[b], y[b] + 1)
                if y.sum() > len(ctx.fids):
                    continue
                f = fitness(y[None, :])[0]
                if f > best_f:
                    best, best_f = y, f
            return best

        # ---------- 双种群演化 ----------
        main = init(self.POP_MAIN)
        aux = init(self.POP_AUX)
        for gen in range(self.GENS):
            # 主种群：选择-交叉-变异 + VNS
            fm = fitness(main)
            new = [vns(main[int(np.argmax(fm))])]
            while len(new) < self.POP_MAIN:
                c = mutate(crossover(selection(main, fm), selection(main, fm)),
                           self.PM_MAIN)
                new.append(c)
            new = np.vstack(new)
            main = np.vstack([main, new])
            main = main[np.argsort(-fitness(main))[: self.POP_MAIN]]

            # 辅助种群：小、变异强，每 5 代与主种群交换精英
            fa = fitness(aux)
            new = np.vstack([mutate(crossover(selection(aux, fa), selection(aux, fa)),
                                    self.PM_AUX) for _ in range(self.POP_AUX)])
            aux = np.vstack([aux, new])
            aux = aux[np.argsort(-fitness(aux))[: self.POP_AUX]]
            if gen % 5 == 0:
                elite = main[:2]
                aux = np.vstack([elite, aux])[: self.POP_AUX]

        best = main[int(np.argmax(fitness(main)))]

        # ---------- 解码：架数 → 就近取机（受 C3 限制） ----------
        order_j = np.argsort(-ctx.values)
        avail = np.ones(len(ctx.fids), dtype=bool)
        for j in order_j:
            k = int(best[j])
            if k <= 0 or not avail.any():
                continue
            d = ctx.dist[:, j].copy()
            d[~avail | ~ctx.c3[:, j]] = np.inf
            idx = np.argsort(d)[: min(k, int(np.isfinite(d).sum()))]
            idx = idx[np.isfinite(d[idx])]
            X[idx, j] = 1
            avail[idx] = False

        return repair_c2(X, ctx.cap, ctx.dist, ctx.c3)
