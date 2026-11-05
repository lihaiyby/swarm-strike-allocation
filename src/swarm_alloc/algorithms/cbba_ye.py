"""叶赞 改进 CBBA（注册名 `cbba_ye`）—— 依据原文实现。

文献
----
叶赞. 基于改进 CBBA 算法的风电场无人机群自主巡检任务分配[J].
数字技术与应用, 2026(5): 238-240.

原文的三处改进（**不是两处**，摘要只提了两处，正文还有价值函数）
------------------------------------------------------------------
1. **风机巡检价值评估函数**（式 1~3）:
   ``W = W0 - (λ·W1 + μ·W2)``，其中 W1 是风速代价、W2 是海拔代价。
2. **任务包双向构建策略**:
   正向以初始机巢为起点、按"价值优先、能耗可控"依次添加任务；
   反向以**所有机巢为潜在终点**反向追溯可覆盖的任务点；
   两类包分别校验"最大航程 / 容量 / 无冲突"后进**候选任务包池**。
3. **基于类别的奖励制度**（式 4~5）:
   先用 **K-means** 把任务点与机巢聚类成若干区域；
   再借人工势场思想，同类任务给奖励、异类无奖励；
   ``G = ζ/2·d*``（同类）或 ``η/2``（异类），总收益 ``SP_i = ΣW_p + G_p``。

原文的五阶段流程：初始化 → 双向任务包构建 → 任务共识 → **最优组合筛选（整数规划）** → 结束判断。

移植到本项目的适配（逐条对照）
------------------------------
| 原文 | 本项目 | 处理 |
|---|---|---|
| 价值函数含风速/海拔代价 | 无风场、无海拔、不建模飞行 | **不移植**，价值直接取场景给定的 `v_j` |
| 每机任务包容量 4 | **每机只能打一个目标**（C1，自杀式） | bundle 退化为单槽；"容量"改指**目标的配额** $m_j$ |
| 机巢 + 航程 45 km | 我方全部部署在内球面，**无航程约束** | 正向池按"离目标最近的我方机"取；反向池按"目标到部署中心距离"取 |
| 整数规划选最优组合 | 同 | **槽展开 + 最大权二分匹配**（`scipy.linear_sum_assignment`） |
| K-means 区域 + 同类奖励 | 目标有空间分布 | **照做**：对目标坐标聚类，同区奖励 |
| 收益量纲 = 能耗(kWh) | 量纲 = 价值(2~10) | ζ/η 按 `v_j` 归一化（见 §参数），否则奖励被量纲淹没 |

**忠实移植的关键一点**：原文的"整数规划确定最优任务包组合"在本项目可以**精确求解**——
把目标 j 展开成 $m_j$ 个槽，第 s 个槽的边际价值为 $v_jp_j(1-p_j)^s$；
这样"带容量的最大权分配"就等价于在槽上的**最大权二分匹配**，用 LSA 精确求解。
这比 greedy 与拍卖都强，也是本实现的核心。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..model import Model
from .base import AlgoBase, register

_NEG = -1e9


def _kmeans(pts: np.ndarray, k: int, iters: int = 30) -> tuple:
    """极简 K-means（最远点初始化，确定性）。返回 (标签, 中心)。"""
    n = len(pts)
    k = max(1, min(k, n))
    centers = [pts[int(np.argmin(pts[:, 0]))]]          # 最左点作首个中心
    for _ in range(k - 1):
        d = np.min(np.linalg.norm(pts[:, None, :] - np.array(centers)[None, :, :],
                                  axis=2), axis=1)
        centers.append(pts[int(np.argmax(d))])          # 最远点
    C = np.array(centers, dtype=float)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dist = np.linalg.norm(pts[:, None, :] - C[None, :, :], axis=2)
        new = np.argmin(dist, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(len(C)):
            m = labels == c
            if m.any():
                C[c] = pts[m].mean(axis=0)
    return labels, C


@register
class CBBAYe(AlgoBase):
    name = "cbba_ye"

    # ---- 原文式(4)的两个增益常量（按 v_j 归一化，见模块文档）
    ZETA: float = 0.30            # ζ：同类增益常量
    ETA: float = 0.05             # η：异类增益常量
    N_REGIONS: int = 0            # 0 = 自动取 min(4, 敌群数)

    # ---- 双向构建的候选池倍数（每个目标保留最近/最远各若干架进池）
    POOL_FACTOR: float = 2.0

    def solve(self, state, model: Model) -> np.ndarray:
        # ---------------- 阶段1 初始化 ----------------
        eids, fids = state.enemy_ids, state.pool_ids
        X = np.zeros((len(fids), len(eids)), dtype=int)
        if not eids or not fids:
            return X

        values, p_kill, epos = state.enemy_matrix()
        fpos = state.pool_matrix()
        self._fpos_cache = fpos
        quotas = np.array([model.quota(p, model.kill_threshold) for p in p_kill])
        eidx = {e: j for j, e in enumerate(eids)}
        group_of = {u: g.id for g in state.groups for u in g.member_ids}
        gmembers = {g.id: [eidx[u] for u in g.member_ids if u in eidx]
                    for g in state.groups}

        # 可行性恢复（与其它算法同口径）：按 select_priority，默认群价值
        demand = {g: int(quotas[ms].sum()) for g, ms in gmembers.items() if ms}
        gvalue_in = {g: float(values[ms].sum()) for g, ms in gmembers.items() if ms}
        kept = model.select_groups(state.groups, demand, len(fids), gvalue_in)
        allowed = {u for g in kept for u in g.member_ids}
        cap = np.array([int(quotas[j]) if eids[j] in allowed else 0
                        for j in range(len(eids))], dtype=int)
        if cap.sum() == 0:
            return X

        dist = np.linalg.norm(fpos[:, None, :] - epos[None, :, :], axis=2)
        dmax = float(dist.max()) + 1e-9

        # ---------------- 阶段2 双向任务包构建 → 候选池 ----------------
        pool = self._build_candidate_pool(dist, cap, len(fids))
        pool &= state.c3_allowed()        # 【批注 B3】组对组：跨群候选直接排除

        # ---------------- 阶段3 基于类别的奖励 ----------------
        bonus = self._category_reward(epos, values, fpos, dist, dmax, dmax)

        # ---------------- 阶段4/5 任务共识 + 最优组合筛选 ----------------
        X = self._assign(pool, cap, values, p_kill, bonus, len(fids))

        # ---------------- 整包裁决（保证 C2） ----------------
        X = self._repair(X, cap, dist, c3=state.c3_allowed())
        return X

    # ------------------------------------------------------------------
    def _build_candidate_pool(self, dist, cap, n_f):
        """双向构建候选池。

        - **正向**（从机巢出发，价值优先）：对每个目标取**距它最近**的若干架我机；
        - **反向**（以机巢为潜在终点）：取**离部署中心最近**的我机（返程最省）。
        两者取并集，即原文"合并后进入候选任务包池"。
        """
        pool = np.zeros((n_f, len(cap)), dtype=bool)
        order_near = np.argsort(dist, axis=0)                 # 每目标：近→远
        base_rank = np.argsort(np.linalg.norm(
            self._fpos_cache, axis=1))                        # 我机：离部署中心近→远
        for j in range(len(cap)):
            if cap[j] == 0:
                continue
            k = max(1, int(np.ceil(self.POOL_FACTOR * cap[j])))
            pool[order_near[:k, j], j] = True                 # 正向
            pool[base_rank[:k], j] = True                     # 反向
        return pool

    # ------------------------------------------------------------------
    def _category_reward(self, epos, values, fpos, dist, dmax, _):
        """式(4)：K-means 分区 + 同类奖励。

        ``G = ζ/2 · v_j · (1 − d_ij/d_max)``  若 i 与 j 同区
        ``G = η/2 · v_j``                     若异区

        原文的 ζ/2·d* 中 d* 是"两点所属类中心的欧氏距离"，OCR 后公式有歧义；
        这里取"无人机到目标所属类中心的距离"，并让奖励随距离**递减**
        （符合人工势场"引力"的语义）。按 v_j 归一化以匹配本项目量纲。
        """
        n_t = len(values)
        k = self.N_REGIONS or int(min(4, max(1, n_t // 12)))
        labels, _ = _kmeans(epos, k)
        # 每架机归属最近的区域中心
        centers = np.array([epos[labels == c].mean(axis=0)
                            for c in range(labels.max() + 1)])
        drone_region = np.argmin(
            np.linalg.norm(fpos[:, None, :] - centers[None, :, :], axis=2), axis=1)

        same = drone_region[:, None] == labels[None, :]
        prox = 1.0 - dist / dmax
        bonus = np.where(same,
                         (self.ZETA / 2.0) * values[None, :] * prox,
                         (self.ETA / 2.0) * values[None, :])
        return bonus

    # ------------------------------------------------------------------
    def _assign(self, pool, cap, values, p_kill, bonus, n_f):
        """槽展开 + 最大权二分匹配（= 原文"整数规划确定最优任务组合"）。"""
        slots = []                       # (target, s)
        for j, c in enumerate(cap):
            for s in range(int(c)):
                slots.append((j, s))
        if not slots:
            return np.zeros((n_f, len(cap)), dtype=int)
        m = len(slots)
        j_arr = np.array([s[0] for s in slots])
        s_arr = np.array([s[1] for s in slots])

        # 第 s 个槽的边际价值 = v_j·p_j·(1-p_j)^s
        marg = values[j_arr] * p_kill[j_arr] * (1.0 - p_kill[j_arr]) ** s_arr
        # 得分矩阵 (n_drones × n_slots)
        score = marg[None, :] + bonus[:, j_arr]
        # 候选池外的对置 -inf
        ok = pool[:, j_arr]
        score = np.where(ok, score, _NEG)

        X = np.zeros((n_f, len(cap)), dtype=int)
        if m == 0:
            return X
        # LSA 求最大权匹配（尽量多配对；不可行的对已被 -inf 排除）
        r, c = linear_sum_assignment(-score)
        for i, s in zip(r, c):
            if score[i, s] > _NEG / 2:
                X[i, j_arr[s]] = 1
        return X

    # ------------------------------------------------------------------
    def _repair(self, X, cap, dist, c3=None):
        """整包裁决：未满配额的目标要么补齐、要么整批撤单（保证 C2）。

        【批注 B3】``c3`` 为 (我机 × 目标) 允许矩阵；补齐时只能用允许的机。
        """
        count = X.sum(axis=0)
        partial = [j for j in range(len(cap)) if 0 < count[j] < cap[j]]
        if not partial:
            return X
        busy = set(np.nonzero(X.any(axis=1))[0].tolist())
        idle = [i for i in range(X.shape[0]) if i not in busy]

        for j in sorted(partial):
            need = int(cap[j] - count[j])
            cand = idle if c3 is None else [i for i in idle if c3[i, j]]
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
