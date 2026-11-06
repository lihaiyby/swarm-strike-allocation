"""曹筱可 享乐博弈分簇 + 层次化匈牙利匹配（注册名 `hedonic`）。

文献
----
曹筱可, 吕金虎, 高源, 等. 规模化异构集群分布式协同任务分配方法[J].
航空学报, 2026, 47(11): 32811.

原文
----
1. **基于享乐博弈的自组织分簇**：通过分布式交互把各类飞行器划分到各任务簇；
2. **隐式共识 + 任务节点分裂机制**，把**匈牙利方法**用于分布式联盟形成，
   完成各簇内飞行器到任务的匹配；
3. **考虑飞行器动力学的代价估计**，实现分配与规划紧耦合。

移植到本项目的适配
------------------
| 原文 | 本项目 | 处理 |
|---|---|---|
| 异构飞行器分簇 | 任务簇 = 敌群（已由感知层给出） | 我机按偏好加入簇（享乐博弈） |
| 隐式共识 / 节点分裂 | 无通信层 | 由"簇容量约束 + 稳定条件"代替 |
| 匈牙利算法 | 同 | 簇内用 `linear_sum_assignment` 做槽展开最优匹配 |
| 动力学代价 | 无动力学模型 | 用距离代价 $\\lambda d_{ij}/R$ 近似 |

享乐博弈的稳定性
----------------
每架机选择使自身效用最大的簇；若某机单独改换簇能提高自身效用，则移动。
迭代至**无人愿意单独改变**（纳什稳定），即享乐博弈的个体稳定划分。
簇的容量约束：一个簇**可接纳的机数 = 该簇所有目标的配额之和**。

【批注 B3】若启用组对组，则每个簇只能由 `c3_allowed()` 允许的机加入。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..model import Model
from ._common import fill_empty_targets, prepare, repair_c2
from .base import AlgoBase, register

_NEG = -1e9


@register
class Hedonic(AlgoBase):
    name = "hedonic"

    LAMBDA_COST: float = 0.35
    MAX_SWITCH: int = 6           # 享乐博弈的最大改簇轮数

    def solve(self, state, model: Model) -> np.ndarray:
        ctx = prepare(state, model)
        X = np.zeros((len(state.pool_ids), len(state.enemy_ids)), dtype=int)
        if ctx is None:
            return X

        eidx = ctx.eidx
        gmembers = {g.id: [eidx[u] for u in g.member_ids if u in eidx]
                    for g in state.groups}
        gmembers = {g: ms for g, ms in gmembers.items() if ms}
        gids = list(gmembers)
        if not gids:
            return X
        cap_g = {g: int(ctx.cap[ms].sum()) for g, ms in gmembers.items()}

        # ---------- 1) 享乐博弈：机 → 簇 ----------
        # 单机对簇的效用 = 该簇内它能拿到的最大（margin − 距离代价）
        util = np.full((len(ctx.fids), len(gids)), _NEG)
        for k, g in enumerate(gids):
            js = gmembers[g]
            u = (ctx.values[js][None, :] * ctx.p_kill[js][None, :]
                 - self.LAMBDA_COST * ctx.dn[:, js])
            u = np.where(ctx.c3[:, js], u, _NEG)
            util[:, k] = u.max(axis=1)

        assign = np.argmax(util, axis=1)          # 每架机先选效用最大的簇
        for _ in range(self.MAX_SWITCH):
            moved = False
            load = np.bincount(assign, minlength=len(gids))
            for i in range(len(ctx.fids)):
                cur = assign[i]
                best_k, best_u = cur, util[i, cur]
                for k in range(len(gids)):
                    if k == cur or load[k] >= cap_g[gids[k]]:
                        continue
                    if util[i, k] > best_u:
                        best_k, best_u = k, util[i, k]
                if best_k != cur:
                    assign[i] = best_k
                    load[cur] -= 1
                    load[best_k] += 1
                    moved = True
            if not moved:
                break

        # ---------- 2) 簇内匈牙利匹配（槽展开） ----------
        for k, g in enumerate(gids):
            js = gmembers[g]
            rows = np.nonzero(assign == k)[0]
            if len(rows) == 0:
                continue
            slots = [(j, s) for j in js for s in range(int(ctx.cap[j]))]
            if not slots:
                continue
            j_arr = np.array([s[0] for s in slots])
            s_arr = np.array([s[1] for s in slots])
            marg = ctx.values[j_arr] * ctx.p_kill[j_arr] * (1 - ctx.p_kill[j_arr]) ** s_arr
            score = marg[None, :] - self.LAMBDA_COST * ctx.dn[rows][:, j_arr]
            ok = ctx.c3[rows][:, j_arr]
            score = np.where(ok, score, _NEG)
            r, c = linear_sum_assignment(-score)
            for a, b in zip(r, c):
                if score[a, b] > _NEG / 2:
                    X[rows[a], j_arr[b]] = 1

        X = fill_empty_targets(X, ctx.cap, ctx.dist, ctx.c3)   # 兜底：整簇未分到机时补齐
        return repair_c2(X, ctx.cap, ctx.dist, ctx.c3)
