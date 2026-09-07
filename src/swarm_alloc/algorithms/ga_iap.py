"""IAP-NSGA-III（刘富樯 2026）复现 + 本场景适配。

适配说明（对应技术报告"改动点"）：
1. 染色体编码：原文 x_ij = 武器 i 向目标 j 发射的弹药数。本场景我方无人机同质、
   自杀式（每架=1发弹），故染色体压缩为**各目标分到的架数向量** c ∈ Z^n，
   c_j ∈ [0, m_j]（遵循原文 Pj,max 边际效益约束的精神：配额 m_j 即封顶，
   避免过杀浪费——这也符合"不设统一硬上限、但算法自限"的适配决策）。
   解码时按"离目标最近"把架数落到具体无人机（我方同质，个体匹配退化为最近邻）。
2. 双目标：f1 = -Σ v_j·P_j(c_j)（打击总收益取负，最小化）；f2 = Σ c_j（成本，最小化）。
3. 保留原文三改进中的两个：DDA-NS 优势度矩阵排序（算法1）、截断均值自适应
   Pc/Pm（式13-17）；粗粒度并行迁移略去（本规模串行已远满足 1s 指标）。
4. 静态模型嵌入滚动时域：每个触发点重跑一次（由仿真主循环驱动）。

参数取原文：Pc∈[0.6,0.9]，Pm∈[0.01,0.1]，B=K=0.3，截断比例 α=0.5。
"""
from __future__ import annotations

import time

import numpy as np

from .base import AlgoBase, register

# ---------- DDA-NS：优势度矩阵非支配排序（原文算法1） ----------

def _ddns(F: np.ndarray) -> list:
    """F: (u × M) 目标矩阵（最小化）。返回分层 [F1, F2, …]，每层为索引数组。"""
    u, h = F.shape
    D = (F[:, None, :] <= F[None, :, :]).sum(axis=2).astype(float)   # 优势度矩阵
    eq = (D == h) & (D.T == h)
    D[eq] = 0
    np.fill_diagonal(D, 0)
    alive = np.ones(u, dtype=bool)
    fronts = []
    while alive.any():
        idx = np.where(alive)[0]
        Q = D[np.ix_(idx, idx)].max(axis=0)
        cur = idx[(Q < h)]
        if len(cur) == 0:
            cur = idx                                                # 兜底
        fronts.append(cur)
        alive[cur] = False
        D[cur, :] = -1
        D[:, cur] = -1
    return fronts


# ---------- 截断均值自适应算子（原文式13-17） ----------

def _truncated_mean(fit: np.ndarray, gen_frac: float, n_f1: int,
                    alpha: float = 0.5, mmax_ratio: float = 0.2) -> float:
    n = len(fit)
    A = n_f1 / n                                                     # 聚集度
    m_eli = int(mmax_ratio * n * gen_frac ** 2 * (1 + alpha * (0.5 - A)))
    m_eli = min(max(m_eli, 0), n - 2)
    return float(np.sort(fit)[: n - m_eli].mean())


def _adaptive_prob(f: float, fmean: float, fmin: float,
                   pmin: float, pmax: float, B: float) -> float:
    """改进 Sigmoid 自适应概率（式16/17，最小化问题：f 越小越优）。"""
    if f < fmean:
        return pmax
    denom = (fmean - fmin) or 1e-12
    return pmax - (pmax - pmin) / (1 + np.exp(B * np.cos((fmean - f) / denom * np.pi)))


class _NSGA3:
    """双目标 NSGA-III 环境选择（参考点关联 + 利基保留）。"""

    def __init__(self, rng, pop_size=100, n_ref_div=2):
        self.rng, self.N = rng, pop_size
        self.refs = np.array([[i / n_ref_div, 1 - i / n_ref_div]
                              for i in range(n_ref_div + 1)])
        self.W = self.refs / np.linalg.norm(self.refs, axis=1, keepdims=True)

    def _assoc(self, Fn: np.ndarray):
        """垂直距离关联：d = |f - (f·w)w|。返回 (最近参考点索引, 距离)。"""
        proj = (Fn @ self.W.T)[:, :, None] * self.W[None, :, :]       # (u, nref, 2)
        d = np.linalg.norm(Fn[:, None, :] - proj, axis=2)
        return d.argmin(axis=1), d.min(axis=1)

    def select(self, F: np.ndarray) -> np.ndarray:
        fronts = _ddns(F)
        chosen: list = []
        last = None
        for fr in fronts:
            if len(chosen) + len(fr) <= self.N:
                chosen.extend(fr)
            else:
                last = fr
                break
        if last is None:
            return np.array(chosen, dtype=int)

        chosen = np.array(chosen, dtype=int)
        need = self.N - len(chosen)
        fmin = F[chosen].min(axis=0) if len(chosen) else F.min(axis=0)
        span = F.max(axis=0) - fmin
        span = np.where(span < 1e-12, 1.0, span)
        Fn = (F - fmin) / span

        r_ch = self._assoc(Fn[chosen])[0] if len(chosen) else np.array([], dtype=int)
        r_last, d_last = self._assoc(Fn[last])
        counts = np.bincount(r_ch, minlength=len(self.refs)).astype(float)
        cand = set(range(len(last)))
        picked = []
        while len(picked) < need and cand:
            j = int(np.argmin(counts))
            members = [c for c in cand if r_last[c] == j]
            if not members:
                counts[j] = np.inf
                continue
            if counts[j] == 0:
                pick = min(members, key=lambda c: d_last[c])
            else:
                pick = int(self.rng.choice(members))
            picked.append(pick)
            cand.remove(pick)
            counts[j] += 1
        return np.concatenate([chosen, last[np.array(picked, dtype=int)]])


@register
class GaIap(AlgoBase):
    name = "ga_iap"

    GEN_MAX = 200
    TIME_BUDGET = 0.8          # 秒，留 1s 上限余量
    POP = 100
    PC_RANGE = (0.6, 0.9)
    PM_RANGE = (0.01, 0.1)
    BK = 0.3

    def solve(self, state, model) -> np.ndarray:
        eids, fids = state.enemy_ids, state.pool_ids
        n, m = len(eids), len(fids)
        X = np.zeros((m, n), dtype=int)
        if not eids or not fids:
            return X

        values, p_kill, epos = state.enemy_matrix()
        eidx = {e: j for j, e in enumerate(eids)}
        rng = np.random.default_rng(eids[0] * 1000 + int(state.t))

        # 可行性恢复（model 职责）：威胁度排序纳入，配额需求受资源池约束
        quotas = np.array([model.quota(p, model.kill_threshold) for p in p_kill])
        demand = {}
        for g in state.groups:
            js = [eidx[uid] for uid in g.member_ids if uid in eidx]
            demand[g.id] = int(quotas[js].sum()) if js else 0
        kept = model.select_groups(state.groups, demand, m)
        allow = np.zeros(n, dtype=bool)
        for g in kept:
            for uid in g.member_ids:
                if uid in eidx:
                    allow[eidx[uid]] = True

        cap = np.where(allow, quotas, 0)

        def evaluate(C: np.ndarray):
            counts = C.astype(float)
            P = 1 - (1 - p_kill)[None, :] ** counts
            f1 = -(P * values).sum(axis=1)
            f2 = counts.sum(axis=1)
            over = np.maximum(f2 - m, 0)                               # 超资源池
            short = ((counts > 0) & (counts < quotas)).sum(axis=1)     # 打了没打够
            pen = 100 * over + 50 * short * values.max()
            return np.column_stack([f1 + pen, f2 + pen]), f1, f2

        ga = _NSGA3(rng, pop_size=self.POP)
        pop = rng.integers(0, cap + 1, size=(self.POP, n))
        pop[rng.random((self.POP, n)) < 0.4] = 0                       # 稀疏化
        pop = np.minimum(pop, cap)

        t0 = time.perf_counter()
        for gen in range(self.GEN_MAX):
            if time.perf_counter() - t0 > self.TIME_BUDGET:
                break
            F, _, _ = evaluate(pop)
            fit = F.sum(axis=1)
            fmean = _truncated_mean(fit, gen / self.GEN_MAX, len(_ddns(F)[0]))
            fmin = fit.min()

            # 均匀交叉（自适应 Pc）
            perm = rng.permutation(self.POP).reshape(-1, 2)
            off = np.empty((len(perm) * 2, n), dtype=int)
            for k, (a, b) in enumerate(perm):
                pc = _adaptive_prob(max(fit[a], fit[b]), fmean, fmin,
                                    *self.PC_RANGE, self.BK)
                ca, cb = pop[a].copy(), pop[b].copy()
                if rng.random() < pc:
                    mask = rng.random(n) < 0.5
                    ca[mask], cb[mask] = pop[b][mask], pop[a][mask]
                off[2 * k], off[2 * k + 1] = ca, cb

            # 变异（自适应 Pm）：随机重设基因
            pm = np.array([_adaptive_prob(fit[i % self.POP], fmean, fmin,
                                          *self.PM_RANGE, self.BK)
                           for i in range(len(off))])
            mut = rng.random(off.shape) < pm[:, None]
            rand_genes = rng.integers(0, cap + 1, size=off.shape)
            off = np.where(mut, rand_genes, off)
            off = np.clip(off, 0, cap)

            keep = ga.select(np.vstack([evaluate(pop)[0], evaluate(off)[0]]))
            pop = np.vstack([pop, off])[keep]

        _, f1, f2 = evaluate(pop)
        best = pop[int(np.argmin(f1 + 1e-6 * f2))]                     # 主目标优先，同分取省兵

        # ---- 解码：架数 → 离目标最近的可用无人机（我方同质，C3 由就近满足） ----
        fpos = state.pool_matrix()
        avail = np.ones(m, dtype=bool)
        for j in np.argsort(-values):
            k = int(best[j])
            if k <= 0 or not avail.any():
                continue
            d = np.linalg.norm(fpos - epos[j], axis=1)
            d[~avail] = np.inf
            idx = np.argsort(d)[: min(k, int(avail.sum()))]
            X[idx, j] = 1
            avail[idx] = False
        return X
