"""商鹏 TR-CBBA：威胁重置捆绑（注册名 `tr_cbba`）。

文献
----
商鹏, 董受全, 刘亿. 基于威胁重置捆绑算法的异构集群动态任务分配方法[J/OL].
指挥控制与仿真, 2026. https://link.cnki.net/urlid/32.1759.TJ.20260603.1044.002

原文
----
以标准 CBBA 为基础，构建四步分布式流程：

1. **受影响单元判定** —— 用"路径风险累积准则"与"时间阈值"快速识别受突发威胁影响的单元；
2. **动态重置范围协商** —— 只对受影响范围重规划，而不是全局重算；
3. **威胁导向捆绑重构** —— 按威胁重新组织任务包，把资源导向高威胁目标；
4. **一致性协商** —— 消解冲突。

目标是解决"突发高威胁目标响应滞后"。

移植到本项目的适配
------------------
| 原文 | 本项目 | 处理 |
|---|---|---|
| 受影响单元判定 | 平台每轮本来就重新感知 | **改为判定"受影响目标群"**：威胁度较上轮跃升超过阈值的群视为受影响 |
| 动态重置范围协商 | 无通信层 | 用"受影响集"决定本轮**只对哪些群做优先捆绑**，其余群按常规顺序 |
| 威胁导向捆绑重构 | 同 | 优先群用**威胁占比**分配兵力预算；非优先群按边际收益填 |
| 一致性协商 | 每轮集中式求解 | 由框架的配额裁决承担 |

**关键差异（必须记录）**：原文的收益来自"局部重规划省下的通信/时间"，
而本项目是集中式、无通信代价、每轮全量重算——**这项优势在本平台无法体现**。
本实现只保留"威胁导向捆绑 + 受影响集优先"这一可移植部分。
"""
from __future__ import annotations

import numpy as np

from ..model import Model
from ._common import marginal, prepare, repair_c2
from .base import AlgoBase, register


@register
class TRCBBA(AlgoBase):
    name = "tr_cbba"

    LAMBDA_COST: float = 0.35
    THREAT_JUMP: float = 0.25     # 威胁度较上轮跃升超过此值 → 受影响

    def __init__(self):
        self._prev_threat: dict = {}

    def solve(self, state, model: Model) -> np.ndarray:
        ctx = prepare(state, model)
        X = np.zeros((len(state.pool_ids), len(state.enemy_ids)), dtype=int)
        if ctx is None:
            return X

        # ---------- 1) 受影响单元判定（威胁跃升） ----------
        cur = {g.id: float(g.threat) for g in state.groups}
        affected = {g for g, t in cur.items()
                    if t - self._prev_threat.get(g, 0.0) > self.THREAT_JUMP}
        self._prev_threat = cur

        # ---------- 2)+3) 威胁导向捆绑：受影响群优先拿兵 ----------
        order = sorted(state.groups, key=lambda g: (g.id not in affected, -g.threat))

        count = np.zeros(len(ctx.eids), dtype=int)
        for g in order:
            js = [ctx.eidx[u] for u in g.member_ids if u in ctx.eidx]
            for _ in range(int(sum(ctx.cap[j] for j in js))):
                used = X.any(axis=1)
                cand = (~used)[:, None] & (count[None, :] < ctx.cap[None, :]) & ctx.c3
                if not cand.any():
                    break
                gain = marginal(ctx.values, ctx.p_kill, count)
                gain = np.where(count < ctx.cap, gain, -np.inf)
                mask = np.zeros(len(ctx.eids), dtype=bool)
                mask[js] = True
                gain = np.where(mask, gain, -np.inf)      # 只在本群内选目标
                dm = np.where(cand, ctx.dn, np.inf)
                bi = np.argmin(dm, axis=0)
                bd = dm[bi, np.arange(len(ctx.eids))]
                ok = np.isfinite(bd) & np.isfinite(gain)
                if not ok.any():
                    break
                u = np.where(ok, gain - self.LAMBDA_COST * bd, -np.inf)
                j = int(np.argmax(u))
                if not np.isfinite(u[j]) or u[j] <= 0:
                    break
                X[int(bi[j]), j] = 1
                count[j] += 1

        return repair_c2(X, ctx.cap, ctx.dist, ctx.c3)
