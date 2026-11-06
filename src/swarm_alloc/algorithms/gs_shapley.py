"""张淳 Gale-Shapley 稳定匹配 + Shapley 值联盟优化（注册名 `gs_shapley`）。

文献
----
张淳, 伍永江, 王雪梅. 一种面向异构无人机群的博弈任务分配算法[J].
小型微型计算机系统, 2026, 47(4).

原文框架（四步）
----------------
1. **任务区域筛选**：按任务需求与空域环境筛出候选无人机集合，压缩问题规模；
2. **偏好计算**：用偏好公式把无人机初步分配到各任务区域；
3. **Gale-Shapley 稳定匹配**：保证初始分配的合理性与稳定性；
4. **联盟形成 + Shapley 值**：用 Shapley 值衡量每架机对联盟收益的**边际贡献**，
   移除低贡献者提升联盟整体收益，并对被移除者做**二次分配**以提高任务完成率。

移植到本项目的适配
------------------
| 原文 | 本项目 | 处理 |
|---|---|---|
| 任务区域筛选 | 目标按敌群分簇 + 距离筛选 | 用 `c3_allowed()` 限制候选对；距离过远（δ 过大）的机不参与 |
| 偏好公式 | 边际毁伤收益 − 距离代价 | 与其它算法同口径 |
| Gale-Shapley | 我机按偏好申请目标；目标按"边际收益"择优，容量 = 配额 | 实现对多（many-to-one）稳定匹配 |
| Shapley 值 | 用**边际贡献**近似（联盟按序加入的边际收益） | 移除贡献低于阈值的机，再二次分配 |

说明：精确 Shapley 值需对所有子联盟求和（指数级），本实现采用**按序列边际贡献**的近似，
并在文档中标注这一偏差。
"""
from __future__ import annotations

import numpy as np

from ..model import Model
from .base import AlgoBase, register


@register
class GSShapley(AlgoBase):
    name = "gs_shapley"

    LAMBDA_COST: float = 0.35
    DELTA_MAX: float = 0.85        # 距离筛选：δ 超过此值的机不参与该目标
    SHAPLEY_KEEP: float = 0.15     # 贡献低于"均值×此系数"的机被移除

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
        dmax = float(dist.max()) + 1e-9
        dn = dist / dmax
        c3 = state.c3_allowed() & (dn <= self.DELTA_MAX)   # 任务区域筛选

        # ---------- 偏好：单机对目标首架次的边际效用 ----------
        base_gain = values * p_kill                         # k=0 时的边际收益
        pref = base_gain[None, :] - self.LAMBDA_COST * dn   # (n_f, n_e)
        pref = np.where(c3 & (cap[None, :] > 0), pref, -np.inf)

        # ---------- Gale-Shapley：我机按偏好降序申请，目标按当前边际收益择优 ----------
        X = self._gale_shapley(pref, cap, values, p_kill, c3)

        # ---------- Shapley 近似：移除低边际贡献者，再二次分配 ----------
        X = self._shapley_prune(X, cap, values, p_kill, dn, c3)

        # ---------- 整包裁决（C2） ----------
        X = self._repair(X, cap, dist, c3)
        return X

    # ------------------------------------------------------------------
    def _gale_shapley(self, pref, cap, values, p_kill, c3):
        """对多（many-to-one）稳定匹配。

        每架未匹配机向尚未拒绝过它的最高偏好目标"申请"；
        目标保留边际收益最高的若干个（容量 = 配额），其余拒绝。
        """
        n_f, n_e = pref.shape
        order = np.argsort(-pref, axis=1)          # 每架机的目标偏好序
        ptr = np.zeros(n_f, dtype=int)
        matched = -np.ones(n_f, dtype=int)
        holders = [[] for _ in range(n_e)]         # 每个目标已接受的机
        free = list(range(n_f))

        while free:
            nxt = []
            for i in free:
                placed = False
                while ptr[i] < n_e:
                    j = int(order[i, ptr[i]])
                    ptr[i] += 1
                    if not np.isfinite(pref[i, j]):
                        continue
                    holders[j].append(i)
                    # 目标择优：按当前边际收益保留前 cap[j] 个
                    if len(holders[j]) > int(cap[j]):
                        k = len(holders[j]) - 1
                        gains = [values[j] * p_kill[j] * (1 - p_kill[j]) ** s
                                 for s in range(len(holders[j]))]
                        gains = np.array(gains)
                        # 近似：按该机对目标的边际收益（此处用偏好值）排序
                        sc = np.array([pref[h, j] + 0.0 for h in holders[j]])
                        worst = int(np.argmin(sc))
                        dropped = holders[j].pop(worst)
                        matched[dropped] = -1
                        nxt.append(dropped)
                    else:
                        matched[i] = j
                    placed = True
                    break
                if not placed and matched[i] < 0:
                    pass                              # 无处可去
            free = [i for i in set(nxt) if matched[i] < 0]

        X = np.zeros((n_f, n_e), dtype=int)
        for j in range(n_e):
            for i in holders[j]:
                X[i, j] = 1
        return X

    # ------------------------------------------------------------------
    def _shapley_prune(self, X, cap, values, p_kill, dn, c3):
        """用"移除后的收益下降"作为边际贡献的近似（Shapley 的一阶近似）。

        贡献低于 (均值 × SHAPLEY_KEEP) 的机被移除，然后尝试二次分配。
        """
        count = X.sum(axis=0).astype(int)
        ii, jj = np.nonzero(X)
        if len(ii) == 0:
            return X
        contrib = np.zeros(len(ii))
        for k, (i, j) in enumerate(zip(ii, jj)):
            if count[j] <= 1:
                contrib[k] = np.inf                    # 独占目标的机不移除
                continue
            P_all = 1 - (1 - p_kill[j]) ** count[j]
            P_wo = 1 - (1 - p_kill[j]) ** (count[j] - 1)
            contrib[k] = values[j] * (P_all - P_wo) - self.LAMBDA_COST * dn[i, j]
        thr = np.mean(contrib[np.isfinite(contrib)]) * self.SHAPLEY_KEEP if np.any(
            np.isfinite(contrib)) else -np.inf
        Y = X.copy()
        removed = []
        for k, (i, j) in enumerate(zip(ii, jj)):
            if contrib[k] < thr:
                Y[i, j] = 0
                count[j] -= 1
                removed.append(i)
        # 二次分配：被移除的机尝试补到仍有缺口的目标（只补最缺的一个）
        use = np.ones(len(Y), dtype=bool)
        use[list(removed)] = False
        for i in removed:
            best = None
            for j in range(len(cap)):
                if count[j] >= cap[j] or not c3[i, j]:
                    continue
                u = (values[j] * p_kill[j] * (1 - p_kill[j]) ** count[j]
                     - self.LAMBDA_COST * dn[i, j])
                if best is None or u > best[0]:
                    best = (u, j)
            if best is not None and best[0] > 0:
                Y[i, best[1]] = 1
                count[best[1]] += 1
                use[i] = True
        return Y

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
