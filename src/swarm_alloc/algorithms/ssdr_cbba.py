"""文凤鹏 SSDR-CBBA：次模优化 + 阴影价格（注册名 `ssdr_cbba`）。

文献
----
文凤鹏, 朱哲, 禹磊, 等. 不确定环境的异构无人机动态任务重分配方法[J/OL].
系统仿真学报, 2026. DOI: 10.16182/j.issn1004731x.joss.26-0149

原文
----
面向海上搜救的不确定环境：任务分为"区域搜索"与"目标救援"，结合目标概率、区域风险、
资源需求建立不确定模型，构建**多约束次模优化模型**；针对任务插入与无人机故障引起的性能退化，
提出**基于局部边际收益的阴影敏感度机制**，将**全局风险、航程与负载映射为影子价格**，
对贪婪决策做**全局修正**，实现毫秒级近似最优重分配。

移植到本项目的适配
------------------
| 原文 | 本项目 | 处理 |
|---|---|---|
| 区域搜索 / 目标救援两类任务 | 只有打击一类 | 不移植 |
| 目标概率 / 区域风险 | 无不确定性建模 | 用**敌群威胁度**代替"区域风险" |
| 多约束次模优化 | 收益函数 $\\sum v_jP_j$ 本身**就是次模的**（凹函数复合） | 保留：贪心有 $(1-1/e)$ 近似保证 |
| 影子价格：全局风险、航程、负载 | 同 | $\\pi = \\alpha\\cdot$风险 $+\\beta\\cdot$航程 $+\\gamma\\cdot$负载 |

**核心机制**：不使用原始边际收益，而用

$$u_{ij}=\\underbrace{v_j\\big(P_j(k_j+1)-P_j(k_j)\\big)}_{\\text{局部边际收益}}
        -\\underbrace{\\pi_i}_{\\text{影子价格}}\\cdot\\tilde d_{ij}$$

其中影子价格随资源消耗线性上升（负载项），使贪婪在后期更保守——这是原文
"对贪婪决策进行全局修正"的移植。
"""
from __future__ import annotations

import numpy as np

from ..model import Model
from ._common import marginal, prepare, repair_c2
from .base import AlgoBase, register


@register
class SSDRCBBA(AlgoBase):
    name = "ssdr_cbba"

    ALPHA_RISK: float = 0.30      # 风险项权重（群威胁度）
    BETA_RANGE: float = 0.35      # 航程项权重
    GAMMA_LOAD: float = 0.60      # 负载项权重（资源越紧，影子价格越高）

    def solve(self, state, model: Model) -> np.ndarray:
        ctx = prepare(state, model)
        X = np.zeros((len(state.pool_ids), len(state.enemy_ids)), dtype=int)
        if ctx is None:
            return X

        eidx = ctx.eidx
        threat_of = {}
        for g in state.groups:
            for u in g.member_ids:
                if u in eidx:
                    threat_of[eidx[u]] = float(g.threat)
        risk = np.array([threat_of.get(j, 0.0) for j in range(len(ctx.eids))])

        total = max(int(ctx.cap.sum()), 1)
        count = np.zeros(len(ctx.eids), dtype=int)

        for _ in range(total):
            used = X.any(axis=1)
            cand = (~used)[:, None] & (count[None, :] < ctx.cap[None, :]) & ctx.c3
            if not cand.any():
                break
            gain = marginal(ctx.values, ctx.p_kill, count)
            gain = np.where(count < ctx.cap, gain, -np.inf)

            dm = np.where(cand, ctx.dn, np.inf)
            bi = np.argmin(dm, axis=0)
            bd = dm[bi, np.arange(len(ctx.eids))]
            ok = np.isfinite(bd) & np.isfinite(gain)
            if not ok.any():
                break

            # 影子价格：风险 + 航程 + 负载（负载项随已用兵力比例上升）
            load = X.sum() / total
            price = (self.ALPHA_RISK * risk
                     + self.BETA_RANGE
                     + self.GAMMA_LOAD * load)
            u = np.where(ok, gain - price * bd, -np.inf)
            j = int(np.argmax(u))
            if not np.isfinite(u[j]) or u[j] <= 0:
                break
            X[int(bi[j]), j] = 1
            count[j] += 1

        return repair_c2(X, ctx.cap, ctx.dist, ctx.c3)
