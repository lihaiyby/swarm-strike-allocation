"""时序仿真（V2-P2）：离散事件 + 飞行时间 + 事件驱动重分配。

为什么新开一条路径
------------------
v1 的 `simulate` 是"4 个固定时刻 + 立即命中"，改动它会**推翻全部既有结果**。
因此 V2 单独实现 `simulate_timed`，两条口径可并行对照：

| | v1 `simulate` | V2 `simulate_timed` |
|---|---|---|
| 决策时刻 | 固定 4 个（0、T/3、2T/3、T） | **事件驱动**：波次到达 / 打击命中 / 战损 / 时间上限 |
| 命中 | 下达即命中 | **飞行时间后命中**（升空 + 水平） |
| 我方部署 | 内球面均匀（可切环形平台） | 环形平台（读 `launch` 配置） |
| 摧毁判定 | `kill_rule` 可切 | 同 |

事件循环
--------
```
t = 0
while t < T:
    t_next = min( 下一次打击到达, 下一波次入场, T )
    推进敌机到 t_next，判定突防
    处理到期的打击（命中结算 + 敌机防御火力）
    若满足重分配条件 → 决策一次（感知 → 可行性筛选 → 算法 → 生成新的在途打击）
```
"""
from __future__ import annotations

import copy
import time
from collections import defaultdict

import numpy as np

from ..perception.dbscan_inc import GroupTracker
from ..perception.topsis import topsis
from ..evaluation.b_metric import group_elimination_rate
from .simulate import BattleResult, _summarize_time
from .state import State

_EPS_T = 1e-9


def _clone(u):
    v = copy.copy(u)
    v.pos = u.pos.copy()
    v.vel = u.vel.copy()
    return v


def _defense_fire_prob(lethality: dict, etype: str) -> float:
    """敌机的防御火力：在途我机被击落的概率。

    `lethality` 的值是 dict（`{mode: point|area, q: 概率, ...}`），取其 `q`。
    本时序模型里"接战"发生在**打击到达时**，故把 `q` 视作
    "这架我机在命中前被击落"的概率。
    """
    leth = lethality.get(etype)
    if not leth:
        return 0.0
    return float(leth.get("q", 0.0) or 0.0)


def simulate_timed(scene, algo, model, scfg, etcfg, fcfg=None,
                   realloc: dict | None = None, seed_offset: int = 0,
                   t_scale: float = 1.0) -> BattleResult:
    """时序仿真主函数。

    参数
    ----
    fcfg : dict | None
        `config/friendly.yaml`；用于取 `launch.climb_height_m / climb_speed_mps`。
        缺省时按 30 m / 20 m/s。
    realloc : dict | None
        重分配触发条件，缺省 ``{on_friendly_lost: True, on_enemy_killed: True,
        on_wave: True, min_interval_s: 1.0, max_decisions: 200}``。
    """
    from ..config import load
    fcfg = fcfg if fcfg is not None else (load("friendly") or {})
    launch = fcfg.get("launch") or {}
    climb_h = float(launch.get("climb_height_m", 30.0))
    climb_v = max(float(launch.get("climb_speed_mps", 20.0)), 1e-9)
    t_climb = climb_h / climb_v

    rc = {"on_friendly_lost": True, "on_enemy_killed": True, "on_wave": True,
          "min_interval_s": 1.0, "max_decisions": 200}
    rc.update(realloc or {})

    rng = np.random.default_rng(scene.seed + int(seed_offset))
    # t_scale：战局时长倍率（实验用；敌机速度不变，只把 T 与其派生的波次时刻同步放大）
    T = float(scene.T) * float(t_scale)
    defense_r = scfg["battlefield"]["defense_radius_m"]
    tracker = GroupTracker(scfg["perception"]["eps_m"], scfg["perception"]["min_samples"])
    lethality = {t["name"]: t["lethality"] for t in etcfg["types"] if t.get("lethality")}
    wave_times = [f * T for f in scfg["waves"]["times_frac"]]

    pool = {fu.id: fu for fu in scene.friendly}          # 未出击的机
    cruise = {fu.id: float(getattr(fu, "speed", 60.0)) for fu in scene.friendly}
    active: dict = {}                                     # 在空敌机
    p_of, v_of, type_of = {}, {}, {}
    P = defaultdict(float)
    hit = defaultdict(bool)
    cum = defaultdict(int)
    inflight: list = []                                   # [(到达时刻, fid, eid, p_ij)]

    res = BattleResult(seed=scene.seed, algo=algo.name, n_enemy=scene.n_enemy)
    res.n_inflight_lost = 0

    n_wave_done = 0
    t = 0.0
    last_decision_t = -1e9
    n_decisions = 0
    pending_wave = True          # t=0 时先入场第一波再决策
    force_decide = True
    killed_since_decision = 0
    lost_since_decision = 0

    def _enter_wave(k: int):
        for u in scene.waves[k].units:
            active[u.id] = _clone(u)
            p_of[u.id], v_of[u.id] = u.p_kill, u.value
            type_of[u.id] = getattr(u, "type_name", "")

    def _advance(dt: float):
        nonlocal lost_since_decision
        for e in active.values():
            e.pos = e.pos + e.vel * dt
        for eid in [i for i, e in active.items() if np.linalg.norm(e.pos) <= defense_r]:
            res.n_leaked += 1
            del active[eid]

    def _resolve(now: float):
        """结算所有到达时刻 ≤ now 的打击。"""
        nonlocal inflight, killed_since_decision, lost_since_decision
        keep = []
        for ta, fid, eid, pij in inflight:
            if ta > now + _EPS_T:
                keep.append((ta, fid, eid, pij))
                continue
            if eid not in active:
                res.n_strike_void += 1
                continue                                  # 目标已被别的架次解决
            e = active[eid]
            # 1) 敌机防御火力：可能先把这架我机打掉
            q = _defense_fire_prob(lethality, type_of.get(eid, ""))
            if q > 0 and rng.random() < q:
                res.n_lost += 1
                lost_since_decision += 1
                continue
            # 2) 命中结算
            res.n_strike_hit += 1
            if model.kill_rule == "monte_carlo":
                if rng.random() < pij:
                    hit[eid] = True
                P[eid] = 1.0 if hit[eid] else 0.0
            else:
                P[eid] = 1.0 - (1.0 - P[eid]) * (1.0 - pij)
            if P[eid] >= model.kill_threshold:
                res.a_gain += v_of[eid] * P[eid]
                res.a_kill += v_of[eid]
                res.n_destroyed += 1
                killed_since_decision += 1
                del active[eid]
        inflight = keep

    def _should_decide(now: float) -> bool:
        nonlocal force_decide
        if force_decide:
            return True
        if n_decisions >= int(rc["max_decisions"]):
            return False
        if now - last_decision_t < float(rc["min_interval_s"]):
            return False
        if rc["on_enemy_killed"] and killed_since_decision > 0:
            return True
        if rc["on_friendly_lost"] and lost_since_decision > 0:
            return True
        return False

    def _decide(now: float):
        nonlocal last_decision_t, n_decisions, force_decide
        nonlocal killed_since_decision, lost_since_decision, inflight
        if not active or not pool:
            return
        eids = list(active.keys())
        fids = list(pool.keys())
        tracker.update(eids, np.vstack([active[i].pos for i in eids]))
        groups = topsis(list(tracker.groups.values()), active)
        st = State(t=now, enemies=dict(active), groups=groups,
                   pool=dict(pool), frozen_ids=[])
        if model.grouping_on:
            from ..model.grouping import build_friendly_groups
            fid2g, g2e, _ = build_friendly_groups(st, model)
            st.friendly_groups, st.group_match = fid2g, {k: v for k, v in g2e.items() if v is not None}

        t0 = time.perf_counter()
        X = algo.solve(st, model)
        _dt = time.perf_counter() - t0
        res.time_s += _dt
        res.time_list.append(_dt)
        n_decisions += 1
        last_decision_t = now
        force_decide = False
        killed_since_decision = 0
        lost_since_decision = 0

        if model.grouping_on:
            from ..model.grouping import project_c3
            X, stt = project_c3(X, st, model, st.friendly_groups, st.group_match)
            res.c3_cross += int(stt["cross"])
            res.c2_dropped += int(stt["dropped"])

        # 【V2-P2】在途去重：在途架数已达配额的目标不再重复派兵
        if inflight:
            cnt = defaultdict(int)
            for _, _, e2, _ in inflight:
                cnt[e2] += 1
            for j, e2 in enumerate(eids):
                if cnt.get(e2, 0) >= int(model.quota(p_of[e2], model.kill_threshold)):
                    X[:, j] = 0

        iis, jjs = np.nonzero(X)
        if len(iis) == 0:
            return
        epos = np.vstack([active[e].pos for e in eids])
        fpos = np.vstack([pool[f].pos for f in fids])
        dmat = np.linalg.norm(fpos[:, None, :] - epos[None, :, :], axis=2)
        ppair = (model.pair_p_matrix(np.array([p_of[e] for e in eids]), dmat)
                 if model.range_decay_on else None)
        for i, j in zip(iis, jjs):
            fid, eid = fids[i], eids[j]
            fu = pool.pop(fid, None)
            if fu is None:
                continue
            res.n_used += 1
            res.sum_dist += float(dmat[i, j])
            pij = float(ppair[i, j]) if ppair is not None else float(p_of[eid])
            v = max(cruise.get(fid, 60.0), 1e-6)
            ta = now + t_climb + float(dmat[i, j]) / v
            if ta <= T + _EPS_T:
                inflight.append((ta, fid, eid, pij))
            else:
                res.n_strike_late += 1
            # 超过战局时长的打击视为"来不及"，机已消耗但无战果
        res.t_delay_wait += 0.0

    # ---------------- 主循环 ----------------
    while True:
        # 1) 处理当前时刻的事件：波次入场 → 打击结算 → 是否决策
        while n_wave_done < len(wave_times) and wave_times[n_wave_done] <= t + _EPS_T:
            _enter_wave(n_wave_done)
            n_wave_done += 1
            if rc["on_wave"]:
                force_decide = True
        _resolve(t)
        if _should_decide(t):
            _decide(t)

        if t >= T - _EPS_T:
            break

        # 2) 找下一个事件时刻（严格大于当前时刻）
        nxt = T
        if n_wave_done < len(wave_times):
            nxt = min(nxt, wave_times[n_wave_done])
        if inflight:
            nxt = min(nxt, min(x[0] for x in inflight))
        if nxt <= t + _EPS_T:                    # 同一时刻已处理完，直接走到 T
            nxt = T
        _advance(nxt - t)
        t = nxt

    # 战局结束
    _resolve(T + 1.0)                        # 结算恰好在 T 到达的打击
    res.n_wasted = sum(1 for (ta, _, _, _) in inflight if ta > T)
    for eid, e in active.items():
        res.a_gain += v_of[eid] * P[eid]

    all_values = {u.id: u.value for w in scene.waves for u in w.units}
    b, num, den = group_elimination_rate(
        P, scene.ground_truth_groups(), all_values, model.kill_threshold)
    res.b_threat, res.b_num, res.b_den = b, num, den
    res.d_cost = res.a_gain / max(res.n_used, 1)
    res.a_net = res.a_gain - (model.distance_cost_c * res.sum_dist
                              if model.distance_cost_on else 0.0)
    res.n_decisions = n_decisions
    _summarize_time(res)
    return res
