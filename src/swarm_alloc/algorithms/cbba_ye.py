"""叶赞 改进 CBBA（注册名 `cbba_ye`）—— 拍卖式 CBBA + 两处改进。

⚠️ **务必先读：本实现的文献保真度说明**
------------------------------------------------
本机没有该文献的正文（AI-HANDOFF 只留了一行摘要：**"任务包双向构建 + 类别奖励"**）。
因此本文件分两部分：

1. **CBBA 主体（标准部分）**：按 Choi / Brunet / How (2009) 的 CBBA 框架实现
   —— 出价（bid）→ 一致性协商（consensus）→ 冲突释放（release）→ 重出价。
   这部分是公开的、可核对的。
2. **两处"改进"（按摘要字面解读，待原文校准，均为可关闭开关）**：
   - `CLASS_BONUS`（类别奖励）：对高价值类别（格斗型 / 范围伤害型，`value ≥ 10`）
     的目标追加 `系数 × v_j` 的奖励，使其在拍卖中被更早抢到；
   - `TWO_WAY`（双向构建）：正向拍卖结束后做**反向释放—重出价**，
     **仅在目标函数变好时才接受**（否则回滚），保证不劣于正向解。
   这两处的具体公式、参数与触发条件**没有原文依据**，是依据本项目语义做的合理猜测。
   **拿到 PDF 后应优先核对这两处。** 另一种可能的解读（"类别奖励"= 补满整个
   敌群时给奖励）保留在 `GROUP_BONUS` 开关里，默认关闭。

本场景的适配（与原文的差异）
--------------------------
- **每个我方无人机只能打一个目标**（自杀式，约束 C1）→ CBBA 的"任务包"退化为
  **单槽包**，算法本质是**带容量约束的并行拍卖**。这是原文的多任务机器人与本项目
  最本质的差异。
- 目标有**配额下限**（C2：要么 0 架，要么 ≥ m_j）→ 拍卖后必须做**整包裁决**：
  未满配额的目标要么补齐、要么整批撤单，否则解不可行。
- 一致性协商在本项目是**集中式模拟**：所有"代理"共用一份出价表，按
  `(出价高 → 距离近 → ID 小)` 决定归属。

参数
----
| 参数 | 默认 | 含义 |
|---|---|---|
| `CLASS_BONUS` | 0.15 | 类别奖励系数（× `v_j`），仅对 `v_j ≥ CLASS_VALUE_FLOOR` |
| `CLASS_VALUE_FLOOR` | 10 | 视为"高价值类别"的价值门槛 |
| `GROUP_BONUS` | 0.0 | 备选解读：补满一个敌群时的奖励系数（× 群价值），默认关闭 |
| `TWO_WAY` | True | 反向释放—重出价（**只在更优时接受**） |
| `MAX_ROUNDS` | 6 | 正向拍卖轮数 |
| `RELEASE_FRAC` | 0.15 | 反向轮释放比例 |
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..model import Model
from .base import AlgoBase, register


@register
class CBBAYe(AlgoBase):
    name = "cbba_ye"

    CLASS_BONUS: float = 0.15
    CLASS_VALUE_FLOOR: float = 10.0
    GROUP_BONUS: float = 0.0
    TWO_WAY: bool = True
    MAX_ROUNDS: int = 6
    RELEASE_FRAC: float = 0.15

    # ------------------------------------------------------------------
    def solve(self, state, model: Model) -> np.ndarray:
        eids, fids = state.enemy_ids, state.pool_ids
        X = np.zeros((len(fids), len(eids)), dtype=int)
        if not eids or not fids:
            return X

        values, p_kill, epos = state.enemy_matrix()
        fpos = state.pool_matrix()
        quotas = np.array([model.quota(p, model.kill_threshold) for p in p_kill])

        eidx = {e: j for j, e in enumerate(eids)}
        group_of = {u: g.id for g in state.groups for u in g.member_ids}
        gmembers = {g.id: [eidx[u] for u in g.member_ids if u in eidx]
                    for g in state.groups}
        gvalue = {g.id: float(sum(values[m] for m in gmembers[g.id]))
                  for g in state.groups}

        # 可行性恢复：资源不足时按威胁度保留高威胁群（与其它算法口径一致）
        demand = {g: int(quotas[ms].sum()) for g, ms in gmembers.items() if ms}
        kept = model.select_groups(state.groups, demand, len(fids))
        allowed = {u for g in kept for u in g.member_ids}
        cap = np.array([int(quotas[j]) if eids[j] in allowed else 0
                        for j in range(len(eids))], dtype=int)
        if cap.sum() == 0:
            return X

        dist = np.linalg.norm(fpos[:, None, :] - epos[None, :, :], axis=2)
        ctx = dict(values=values, p_kill=p_kill, quotas=quotas, dist=dist,
                   group_of=group_of, gmembers=gmembers, gvalue=gvalue, cap=cap)

        # ---- 正向拍卖 ----
        count = np.zeros(len(eids), dtype=int)
        free = set(range(len(fids)))
        for _ in range(self.MAX_ROUNDS):
            if not free:
                break
            free = self._auction(X, free, count, ctx, eids)
            free |= self._repair(X, count, ctx)      # 整包裁决，返回被撤单的机

        # ---- 反向释放—重出价（只在更优时接受）----
        if self.TWO_WAY:
            base_obj = self._objective(X, values, p_kill)
            X2, cnt2 = X.copy(), count.copy()
            freed = self._release_lowest(X2, cnt2, ctx)
            for _ in range(self.MAX_ROUNDS):
                if not freed:
                    break
                freed = self._auction(X2, freed, cnt2, ctx, eids)
                freed |= self._repair(X2, cnt2, ctx)
            if self._objective(X2, values, p_kill) > base_obj + 1e-9:
                X, count = X2, cnt2

        return X

    # ------------------------------------------------------------------
    # 出价
    # ------------------------------------------------------------------
    def _marginal(self, j, count, values, p_kill):
        k = int(count[j])
        return float(values[j] * p_kill[j] * (1.0 - p_kill[j]) ** k)

    def _bonus(self, j, count, ctx, eids):
        """类别奖励 + 可选的整群完成奖励。"""
        b = 0.0
        if self.CLASS_BONUS > 0 and ctx["values"][j] >= self.CLASS_VALUE_FLOOR:
            b += self.CLASS_BONUS * float(ctx["values"][j])
        if self.GROUP_BONUS > 0:
            gid = ctx["group_of"].get(eids[j])
            if gid is not None:
                mem = ctx["gmembers"].get(gid, [])
                if all(int(count[m]) + (1 if m == j else 0) >= ctx["cap"][m]
                       for m in mem):
                    b += self.GROUP_BONUS * ctx["gvalue"].get(gid, 0.0)
        return b

    def _auction(self, X, free, count, ctx, eids):
        """一轮并行拍卖。返回本轮没抢到任何目标的机。

        进入时**从 X 重算 count 并剔除已分配的机**，避免轮次间的状态漂移
        （否则可能出现同一架机被指派两次，破坏 C1）。
        """
        cap, dist = ctx["cap"], ctx["dist"]
        count[:] = X.sum(axis=0)
        free = {i for i in free if X[i].sum() == 0}

        by_target = defaultdict(list)
        for i in sorted(free):
            best = None
            for j in range(len(eids)):
                if cap[j] == 0 or count[j] >= cap[j]:
                    continue
                s = self._marginal(j, count, ctx["values"], ctx["p_kill"])
                if s <= 0:
                    continue
                s += self._bonus(j, count, ctx, eids)
                key = (s, -float(dist[i, j]), -i)     # 出价高→距离近→ID 小
                if best is None or key > best[0]:
                    best = (key, j)
            if best is not None:
                by_target[best[1]].append((best[0], i))

        nxt = set()
        for j, cands in by_target.items():
            cands.sort(reverse=True)
            take = max(0, int(cap[j] - count[j]))
            for _, i in cands[:take]:
                X[i, j] = 1
                count[j] += 1
            for _, i in cands[take:]:
                nxt.add(i)
        return nxt

    # ------------------------------------------------------------------
    # 整包裁决（保证 C2）
    # ------------------------------------------------------------------
    def _repair(self, X, count, ctx):
        """未满配额的目标：要么补齐，要么整批撤单。返回被撤下的机。"""
        cap, dist = ctx["cap"], ctx["dist"]
        count[:] = X.sum(axis=0)          # 与 X 对齐，避免计数漂移
        freed = set()
        partial = [j for j in range(len(cap)) if 0 < count[j] < cap[j]]
        if not partial:
            return freed
        # 已用掉的机（含部分目标的机）
        busy = set(np.nonzero(X.any(axis=1))[0].tolist())
        idle = [i for i in range(X.shape[0]) if i not in busy]

        for j in sorted(partial, key=lambda x: -float(ctx["dist"][:, x].min())):
            need = int(cap[j] - count[j])
            if len(idle) >= need:
                # 补齐：取最近的空闲机
                pick = sorted(idle, key=lambda i: float(dist[i, j]))[:need]
                for i in pick:
                    X[i, j] = 1
                    count[j] += 1
                    idle.remove(i)
            else:
                # 撤单
                for i in np.nonzero(X[:, j])[0]:
                    X[i, j] = 0
                    count[j] -= 1
                    idle.append(int(i))
                    freed.add(int(i))
        return freed

    # ------------------------------------------------------------------
    def _release_lowest(self, X, count, ctx):
        """撤掉边际收益最低的若干架，返回这些机的集合。"""
        ii, jj = np.nonzero(X)
        if len(ii) == 0:
            return set()
        gains = np.array([self._marginal(j, count, ctx["values"], ctx["p_kill"])
                          for j in jj])
        n_rel = max(1, int(len(ii) * self.RELEASE_FRAC))
        freed = set()
        for k in np.argsort(gains)[:n_rel]:
            i, j = int(ii[k]), int(jj[k])
            X[i, j] = 0
            count[j] -= 1
            freed.add(i)
        return freed

    # ------------------------------------------------------------------
    def _objective(self, X, values, p_kill) -> float:
        k = X.sum(axis=0)
        return float(np.sum(values * (1.0 - (1.0 - p_kill) ** k)))
