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


def _intercept_time(p0, e0, ev, v: float) -> float:
    """直线迎头拦截所需时间（秒）。

    敌我**同时在飞**：我方从 ``p0`` 以速度 ``v`` 直线飞向拦截点，敌机从 ``e0``
    以恒速矢量 ``ev`` 继续飞。令两者在 $t$ 时刻相遇：

        |D + E·t| = v·t,  D = e0 − p0, E = ev
      → (|E|² − v²)·t² + 2(D·E)·t + |D|² = 0

    取最小的正根。若 $v \\le |E|$ 且敌机在远离，可能无正根，返回 ``inf``。

    这与"把目标当成静止、用 d/v 估时"的差别很大：敌人朝我方飞来时，
    相对接近速度是 $v+|E|$，时间约减半。
    """
    D = np.asarray(e0, dtype=float) - np.asarray(p0, dtype=float)
    E = np.asarray(ev, dtype=float)
    a = float(E @ E) - v * v
    b = 2.0 * float(D @ E)
    c = float(D @ D)
    if abs(a) < 1e-9:
        return -c / b if abs(b) > 1e-12 else float("inf")
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return float("inf")
    root = np.sqrt(disc)
    cands = [(-b - root) / (2 * a), (-b + root) / (2 * a)]
    pos = [t for t in cands if t > 1e-9 and np.isfinite(t)]
    return float(min(pos)) if pos else float("inf")


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
                   t_scale: float = 1.0, end_mode: str = "engagement",
                   t_cap_scale: float = 5.0) -> BattleResult:
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
        """推进敌机；判定是否在本区间内越过防线。

        不能用"推进后 |pos| ≤ R"来判突防——**大步长时敌机会穿过球心跑到另一侧**，
        此时 |pos| 又会变大，导致漏判（实测会让 70 架敌机凭空消失）。
        改为**射线与球面求交**：沿速度方向解 $|p+vt|=R$，若交点时刻落在
        $(0, dt]$ 内即判突防。
        """
        nonlocal lost_since_decision
        if dt <= 0:
            return
        for eid in list(active.keys()):
            e = active[eid]
            p = e.pos
            v = e.vel
            if float(p @ p) <= defense_r * defense_r:
                res.n_leaked += 1
                del active[eid]
                continue
            a = float(v @ v)
            if a < 1e-12:
                continue
            b = 2.0 * float(p @ v)
            c = float(p @ p) - defense_r * defense_r
            disc = b * b - 4.0 * a * c
            if disc < 0:
                e.pos = p + v * dt
                continue
            root = np.sqrt(disc)
            t_hit = (-b - root) / (2.0 * a)
            if 0.0 < t_hit <= dt + _EPS_T:
                res.n_leaked += 1
                del active[eid]
            else:
                e.pos = p + v * dt

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
            if fid not in pool:
                continue
            v = max(cruise.get(fid, 60.0), 1e-6)
            # 【修正 1】敌我同时在飞 → 按直线迎头拦截解相遇时间（不再把目标当静止）
            t_int = _intercept_time(fpos[i], epos[j], active[eid].vel, v)
            ta = now + t_climb + t_int
            # 【规则】可行性判定：拦截必须发生在**敌机越过防线之前**
            #   （end_mode="engagement" 的主判据；不再用全局 T 当硬截止）
            if not np.isfinite(t_int):
                res.n_strike_late += 1
                continue
            r_e = float(np.linalg.norm(epos[j]))
            u_e = float(np.linalg.norm(active[eid].vel)) or 1e-9
            t_to_def = max((r_e - defense_r), 0.0) / u_e
            if t_climb + t_int > t_to_def + _EPS_T:
                res.n_strike_late += 1          # 赶在敌机突防之前拦不住
                continue
            if end_mode == "fixed_t" and ta > T + _EPS_T:
                res.n_strike_late += 1
                continue
            pool.pop(fid, None)
            res.n_used += 1
            res.sum_dist += float(dmat[i, j])
            pij = float(ppair[i, j]) if ppair is not None else float(p_of[eid])
            inflight.append((ta, fid, eid, pij))
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

        if end_mode == "fixed_t":
            if t >= T - _EPS_T:
                break
            hard_end = T
        else:
            # engagement：战局在"敌机全部了结（被歼灭或突防）且无在途打击"时结束
            if not active and not inflight:
                break
            hard_end = T * float(t_cap_scale)
            if t >= hard_end - _EPS_T:
                break

        # 2) 找下一个事件时刻（严格大于当前时刻）
        nxt = hard_end
        if n_wave_done < len(wave_times):
            nxt = min(nxt, wave_times[n_wave_done])
        if inflight:
            nxt = min(nxt, min(x[0] for x in inflight))
        if nxt <= t + _EPS_T:                    # 同一时刻已处理完，直接走到收尾时刻
            nxt = hard_end
        _advance(nxt - t)
        t = nxt

    # 战局结束：结算所有仍需结算的在途打击（覆盖"时间到了但飞机还在飞"的情形）
    _resolve(t + 1e9)
    res.n_wasted = sum(1 for (ta, _, _, _) in inflight if ta > t)
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
