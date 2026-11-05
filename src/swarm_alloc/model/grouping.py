"""【批注 B3】我方分组 + 组对组对抗（强制 C3）。

背景
----
`方案04-Long-修改.docx` 正文："根据敌群分组，我方也分出相同数量的组别进行对抗（约束？）"；
江龙批注 #6："我方也可以考虑分组，采用组对组的对抗"；2026-11-05 明确为"**需要做**"。

此前 C3 一直是**死代码**：`Model.check` 里有 C3 分支，但 `group_of_friendly` 全工程无构造，
且 `Model.check` 从未被主循环调用。

三种分组方式（按任务要求全部实现）
----------------------------------
| mode | 名称 | 做法 |
|---|---|---|
| `even` | 按敌群数量均分 | 我机按 ID 排序后切成 G 份（每份约 30~50 架），再按"组心最近"与敌群配对 |
| `sector` | 按空间方位分 | 按方位角排序后按角度均分成 G 个扇区，再按"组心最近"与敌群配对 |
| `nearest` | 就近组队 | 每架机直接归属**最近**的敌群质心；组大小自然不均衡，无需再配对 |

配对规则（`even` / `sector` 需要）
---------------------------------
G 个机群与 G 个敌群做**一对一指派**，用"组心距离"做贪心匹配（最近者优先）。
`nearest` 模式的分组本身就定义了配对，不再额外匹配。

强制执行（投影 + 修复）
----------------------
框架在 `algo.solve` 返回后做两级处理，保证任何算法都能在 C3 下得到**可行解**：

1. **投影**：把跨群的指派清零；
2. **修复**：对"清零后不足配额"的目标，从**允许该目标的空闲机**里按距离补足；
   若补不足，整批撤单（与既有 C2 口径一致）。

这样算法本身不必感知分组也能跑——但**不按组规划的算法会损失更多**，这正是要测的东西。
"""
from __future__ import annotations

import numpy as np


def _centroids(ids, pos_of):
    return {gid: np.mean([pos_of[u] for u in members], axis=0)
            for gid, members in ids.items() if members}


def build_friendly_groups(state, model) -> tuple:
    """构造我机分组并与敌群配对。

    返回 ``(fid_to_gid, gid_to_enemy_gid, pos_of)``：
    - ``fid_to_gid``：我机 id → 我方组号（0..G-1）
    - ``gid_to_enemy_gid``：我方组号 → 敌群 id
    """
    enemy_groups = {g.id: [u for u in g.member_ids if u in state.enemies]
                    for g in state.groups}
    enemy_groups = {k: v for k, v in enemy_groups.items() if v}
    G = len(enemy_groups)
    fids = list(state.pool_ids)
    if G == 0 or not fids:
        return {}, {}, {}

    fpos = state.pool_matrix()
    pos_of = {fid: fpos[i] for i, fid in enumerate(fids)}
    epos = {gid: np.mean([state.enemies[u].pos for u in ms], axis=0)
            for gid, ms in enemy_groups.items()}
    mode = model.grouping_mode

    # ---------- 1) 构造我机分组 ----------
    if mode == "nearest":
        # 每架机直接归属最近的敌群质心 → 分组即配对，无需再匹配
        order = list(enemy_groups)
        enemy_to_gid = {g: i for i, g in enumerate(order)}
        gid_to_enemy = {i: g for i, g in enumerate(order)}
        fid_to_gid = {}
        for fid in fids:
            d = [np.linalg.norm(pos_of[fid] - epos[g]) for g in order]
            fid_to_gid[fid] = enemy_to_gid[order[int(np.argmin(d))]]
        return fid_to_gid, gid_to_enemy, pos_of

    # even / sector：先切成 G 份
    if mode == "sector":
        ang = np.arctan2(fpos[:, 1], fpos[:, 0])
        order_idx = np.argsort(ang)
    else:                                     # even
        order_idx = np.arange(len(fids))
    chunks = np.array_split(order_idx, G)
    fid_to_gid, group_cent = {}, {}
    for k, idx in enumerate(chunks):
        members = [fids[i] for i in idx]
        for fid in members:
            fid_to_gid[fid] = k
        group_cent[k] = np.mean([pos_of[f] for f in members], axis=0) if members else None

    # ---------- 2) 机群 ↔ 敌群 一对一指派（组心最近优先） ----------
    pairs = []
    for k, c in group_cent.items():
        if c is None:
            continue
        for g, e in epos.items():
            pairs.append((float(np.linalg.norm(c - e)), k, g))
    pairs.sort()
    used_k, used_g, gid_to_enemy = set(), set(), {}
    for _, k, g in pairs:
        if k in used_k or g in used_g:
            continue
        gid_to_enemy[k] = g
        used_k.add(k)
        used_g.add(g)
    # 未配上的组（数量极少）挂到剩余敌群
    rest = [g for g in epos if g not in used_g]
    for k in group_cent:
        if k not in gid_to_enemy:
            gid_to_enemy[k] = rest.pop() if rest else None
    return fid_to_gid, gid_to_enemy, pos_of


def project_c3(X: np.ndarray, state, model, fid_to_gid: dict,
               gid_to_enemy: dict, repair: bool = True) -> tuple:
    """把 X 投影到 C3 可行域并修复 C2。

    返回 ``(X_new, stats)``，stats 含投影前后的跨群违规与撤单数，用于诊断。
    """
    fids = state.pool_ids
    eids = state.enemy_ids
    enemy_group_of = {u: g.id for g in state.groups for u in g.member_ids}
    values, p_kill, epos = state.enemy_matrix()
    quotas = np.array([model.quota(p, model.kill_threshold) for p in p_kill])

    allowed = np.zeros_like(X, dtype=bool)
    for i, fid in enumerate(fids):
        g = fid_to_gid.get(fid)
        tgt = gid_to_enemy.get(g) if g is not None else None
        if tgt is None:
            continue
        for j, eid in enumerate(eids):
            if enemy_group_of.get(eid) == tgt:
                allowed[i, j] = True

    cross = int(((X > 0) & ~allowed).sum())
    Y = np.where(allowed, X, 0).astype(int)

    if not repair:
        return Y, dict(cross=cross, dropped=0)

    # ---- C2 修复：不足配额就补，补不齐就整批撤单 ----
    fpos = state.pool_matrix()
    dist = np.linalg.norm(fpos[:, None, :] - epos[None, :, :], axis=2)
    count = Y.sum(axis=0)
    dropped = 0
    for j in range(len(eids)):
        if count[j] == 0 or count[j] >= quotas[j]:
            continue
        need = int(quotas[j] - count[j])
        cand = [i for i in range(len(fids)) if allowed[i, j] and Y[i].sum() == 0]
        if len(cand) >= need:
            pick = sorted(cand, key=lambda i: float(dist[i, j]))[:need]
            for i in pick:
                Y[i, j] = 1
                count[j] += 1
        else:
            for i in np.nonzero(Y[:, j])[0]:
                Y[i, j] = 0
            count[j] = 0
            dropped += 1
    return Y, dict(cross=cross, dropped=dropped)
