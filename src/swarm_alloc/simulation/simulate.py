"""仿真主循环（事件驱动）。

时序（方案04-v3）：
- 触发点：波次时刻 0、T/3、2T/3 s（新敌机出现），以及战局结束 T（最终结算）；
- 每个触发点：推进敌机位置 → 突防判定（越过 700m 防线即漏网）→ 新波次入场
  → 增量 DBSCAN 分组 → TOPSIS 威胁评估 → 可行性恢复 → 算法求解（计时）
  → 结算（期望值毁伤 + 战损抽样）→ 若发生战损且场上仍有敌机，追加一轮重分配；
- 终止：敌机耗尽 / 资源池耗尽 / 到达 T。

口径：毁伤判定为期望值口径（P_j≥0.9 即摧毁）；战损抽样用场景种子 rng，逐次可复现。
"""
from __future__ import annotations

import copy
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..evaluation.b_metric import group_elimination_rate
from ..perception.dbscan_inc import GroupTracker
from ..perception.topsis import topsis
from .state import State
from .settle import battle_damage


@dataclass
class BattleResult:
    seed: int
    algo: str
    a_gain: float = 0.0
    b_threat: float = 0.0          # 威胁消除率（真值团 + 价值加权，见 evaluation/b_metric.py）
    b_num: float = 0.0             # b 的分子：被全歼的真值团价值之和
    b_den: float = 0.0             # b 的分母：场景总价值（常数，与算法无关）
    d_cost: float = 0.0
    time_s: float = 0.0
    n_enemy: int = 0
    n_destroyed: int = 0
    n_leaked: int = 0
    n_lost: int = 0
    n_used: int = 0
    history: list = field(default_factory=list)     # 每触发点一行 dict
    snapshots: list = field(default_factory=list)   # 每触发点 (t, active_pos, X_lines, pool_pos)


def _clone(u):
    v = copy.copy(u)
    v.pos = u.pos.copy()
    v.vel = u.vel.copy()
    return v


def simulate(scene, algo, model, scfg, etcfg, realloc_on_loss: bool = True) -> BattleResult:
    rng = np.random.default_rng(scene.seed)
    tracker = GroupTracker(scfg["perception"]["eps_m"], scfg["perception"]["min_samples"])
    lethality = {t["name"]: t["lethality"] for t in etcfg["types"] if t.get("lethality")}
    defense_r = scfg["battlefield"]["defense_radius_m"]

    pool = {fu.id: fu for fu in scene.friendly}
    active = {}                                   # enemy_id -> EnemyUnit（副本，位置随时间推进）
    cum = defaultdict(int)                        # enemy_id -> 累计被分配架数
    P = defaultdict(float)                        # enemy_id -> 当前累积毁伤
    p_of = {}                                     # enemy_id -> p_kill
    v_of = {}                                     # enemy_id -> value

    res = BattleResult(seed=scene.seed, algo=algo.name, n_enemy=scene.n_enemy)
    prev_t = 0.0

    triggers = [f * scene.T for f in scfg["waves"]["times_frac"]] + [scene.T]
    for ti, t in enumerate(triggers):
        # 1) 推进 + 突防
        dt = t - prev_t
        prev_t = t
        for e in active.values():
            e.pos = e.pos + e.vel * dt
        for eid in [i for i, e in active.items() if np.linalg.norm(e.pos) <= defense_r]:
            res.n_leaked += 1
            del active[eid]

        # 2) 新波次入场
        if ti < len(scene.waves):
            for u in scene.waves[ti].units:
                active[u.id] = _clone(u)
                p_of[u.id], v_of[u.id] = u.p_kill, u.value

        if not active or not pool:
            res.history.append(dict(t=t, n_field=len(active), n_pool=len(pool),
                                    n_destroyed=res.n_destroyed, n_leaked=res.n_leaked,
                                    n_lost=res.n_lost, n_used=res.n_used))
            continue

        # 3) 感知 + 分配（可因战损追加一轮）
        round_no, do_round = 0, True
        while do_round and active and pool:
            do_round = False
            eids = sorted(active)
            epos = np.vstack([active[i].pos for i in eids])
            tracker.update(eids, epos)
            groups = topsis(list(tracker.groups.values()), active)

            state = State(t=t, enemies=dict(active), groups=groups,
                          pool=dict(pool), frozen_ids=[])
            t0 = time.perf_counter()
            X = algo.solve(state, model)
            res.time_s += time.perf_counter() - t0

            fids = state.pool_ids
            ii, jj = np.nonzero(X)
            engaged = {}
            for i, j in zip(ii, jj):
                fid, eid = fids[i], eids[j]
                del pool[fid]                     # 自杀式：下达指令即消耗
                res.n_used += 1
                cum[eid] += 1
                engaged.setdefault(eid, active[eid])
                P[eid] = model.damage(p_of[eid], cum[eid])

            # 4) 结算：期望值毁伤
            for eid in list(engaged):
                if P[eid] >= model.kill_threshold:
                    res.a_gain += v_of[eid] * P[eid]
                    res.n_destroyed += 1
                    del active[eid]

            # 战损抽样 + 触发追加分配
            lost = battle_damage(rng, list(engaged.values()), pool, lethality)
            for fid in lost:
                pool.pop(fid, None)
            res.n_lost += len(lost)
            if realloc_on_loss and lost and active and pool:
                do_round = True
                round_no += 1

            if ti < len(scene.waves):             # 快照（仅波次触发点）
                res.snapshots.append((t, np.array([e.pos for e in active.values()]).reshape(-1, 3),
                                      X, np.array([f.pos for f in pool.values()]).reshape(-1, 3),
                                      [e.pos for e in engaged.values()]))

            res.history.append(dict(t=t, n_field=len(active), n_pool=len(pool),
                                    n_destroyed=res.n_destroyed, n_leaked=res.n_leaked,
                                    n_lost=res.n_lost, n_used=res.n_used))

        if not pool:
            break
        # 场上暂时无敌机但还有后续波次：继续等待下一触发点

    # 战局结束：场上剩余敌机按期望值计入部分收益
    for eid, e in active.items():
        res.a_gain += v_of[eid] * P[eid]

    # b：真值团 + 价值加权，**全程只算一次**（不分回合，不依赖感知层）。
    # 分母 = 场景总价值，对任何算法都是同一个常数。
    all_values = {u.id: u.value for w in scene.waves for u in w.units}
    b, num, den = group_elimination_rate(
        P, scene.ground_truth_groups(), all_values, model.kill_threshold)
    res.b_threat, res.b_num, res.b_den = b, num, den
    res.d_cost = res.a_gain / max(res.n_used, 1)
    return res
