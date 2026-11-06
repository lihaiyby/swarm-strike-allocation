"""算法共用的小工具：态势解包、C3 允许矩阵、C2 整包裁决。

动机：A 轨复现的算法越来越多（8 个），`select_groups → cap → c3_allowed → 整包裁决`
这套脚手架在每个文件里重复一遍既啰嗦又容易写错。这里集中实现一次。
"""
from __future__ import annotations

import numpy as np


class Ctx:
    """一次 solve 调用所需的全部静态信息。"""

    __slots__ = ("eids", "fids", "values", "p_kill", "quotas", "cap",
                 "c3", "dist", "dn", "epos", "fpos", "eidx")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def prepare(state, model) -> Ctx | None:
    """把态势解包成算法好用的形式；若当前无可打目标返回 None。"""
    eids, fids = state.enemy_ids, state.pool_ids
    if not eids or not fids:
        return None

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
        return None

    dist = np.linalg.norm(fpos[:, None, :] - epos[None, :, :], axis=2)
    dn = dist / (np.linalg.norm(epos, axis=1).max() + 1e-9)
    return Ctx(eids=eids, fids=fids, values=values, p_kill=p_kill,
               quotas=quotas, cap=cap, c3=state.c3_allowed(), dist=dist,
               dn=dn, epos=epos, fpos=fpos, eidx=eidx)


def repair_c2(X, cap, dist, c3):
    """整包裁决：未满配额的目标要么补齐、要么整批撤单（保证 C2）。"""
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


def fill_empty_targets(X, cap, dist, c3):
    """把"一架都没分到"但有配额的目标补满（用空闲且允许的机，就近取）。

    用于分簇类算法（如 hedonic）：簇内匹配后可能整簇没分到机，需要兜底。
    """
    count = X.sum(axis=0)
    busy = set(np.nonzero(X.any(axis=1))[0].tolist())
    idle = [i for i in range(X.shape[0]) if i not in busy]
    for j in np.argsort(-cap):                    # 需求大的先补
        if cap[j] == 0 or count[j] > 0:
            continue
        need = int(cap[j])
        cand = [i for i in idle if c3[i, j]]
        if len(cand) < need:
            continue
        for i in sorted(cand, key=lambda i: float(dist[i, j]))[:need]:
            X[i, j] = 1
            idle.remove(i)
            count[j] += 1
    return X


def marginal(values, p_kill, count):
    """第 count[j]+1 架对目标 j 的边际期望收益。"""
    return values * ((1.0 - (1.0 - p_kill) ** (count + 1))
                     - (1.0 - (1.0 - p_kill) ** count))
