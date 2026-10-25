"""oracle（P1.1）的正确性测试：与暴力枚举对齐 + 上界性质。"""
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm_alloc.algorithms import greedy  # noqa: F401
from swarm_alloc.algorithms.base import get
from swarm_alloc.config import ROOT, load
from swarm_alloc.evaluation.oracle import (_knapsack_a, _knapsack_b,
                                           scene_upper_bound)
from swarm_alloc.model import Model
from swarm_alloc.simulation.simulate import simulate

scfg, etcfg = load("scenario"), load("enemy_types")
model = Model(load("eval"))

with open(ROOT / "data" / "scenes.pkl", "rb") as f:
    SCENES = pickle.load(f)


def _brute_a(values, p_kill, quotas, budget, extra_cap):
    best = -1.0
    opts = []
    for j in range(len(values)):
        q = int(quotas[j])
        opts.append([0] + list(range(q, min(budget, q + extra_cap) + 1)))
    for combo in itertools.product(*opts):
        if sum(combo) > budget:
            continue
        val = sum(v * (1 - (1 - p) ** k)
                  for v, p, k in zip(values, p_kill, combo))
        best = max(best, val)
    return best


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_knapsack_a_matches_brute_force(seed):
    """多选择背包必须与暴力枚举完全一致（小规模）。"""
    rng = np.random.default_rng(seed)
    n = 5
    values = rng.choice([2, 5, 10], size=n).astype(float)
    p_kill = rng.choice([0.4, 0.7, 0.9], size=n)
    quotas = np.array([int(np.ceil(np.log(0.1) / np.log(1 - p))) for p in p_kill])
    budget = 14
    cap = 4
    got, counts = _knapsack_a(values, p_kill, quotas, budget, extra_cap=cap)
    want = _brute_a(values, p_kill, quotas, budget, cap)
    assert got == pytest.approx(want, abs=1e-9)
    assert counts.sum() <= budget
    # 配额下限：非零架数必须 ≥ 配额
    for c, q in zip(counts, quotas):
        assert c == 0 or c >= q


def test_knapsack_b_matches_brute_force():
    """0/1 背包（群级）与暴力枚举一致。"""
    gv = np.array([10.0, 30.0, 20.0, 40.0])
    gc = np.array([3, 7, 4, 9])
    budget = 14
    got, chosen = _knapsack_b(gv, gc, budget)
    best = 0.0
    for mask in range(1 << len(gv)):
        sel = [i for i in range(len(gv)) if mask >> i & 1]
        if sum(gc[i] for i in sel) <= budget:
            best = max(best, sum(gv[i] for i in sel))
    assert got == pytest.approx(best)
    assert sum(gc[i] for i in chosen) <= budget
    assert sum(gv[i] for i in chosen) == pytest.approx(got)


@pytest.mark.parametrize("idx", [0, 12, 33, 49])
def test_a_opt_dominates_actual_algorithms(idx):
    """上界性质：任何算法的 a 都不得超过 a_opt。"""
    s = SCENES[idx]
    ob = scene_upper_bound(s, model)
    r_g = simulate(s, get("greedy"), model, scfg, etcfg)
    assert r_g.a_gain <= ob.a_opt + 1e-6
    assert ob.a_opt > 0


@pytest.mark.parametrize("idx", [0, 12])
def test_b_opt_dominates_actual_b(idx):
    """上界性质：任何算法的 b 都不得超过 b_opt。"""
    s = SCENES[idx]
    ob = scene_upper_bound(s, model)
    r_g = simulate(s, get("greedy"), model, scfg, etcfg)
    assert r_g.b_threat <= ob.b_opt + 1e-9
    assert 0.0 <= ob.b_used <= 1.0


def test_budget_is_fleet_size():
    s = SCENES[0]
    ob = scene_upper_bound(s, model)
    assert ob.budget == len(s.friendly)
    assert ob.used <= ob.budget


# ---------------------------------------------------------------- 近视最优参照策略

def _state_at_wave0(s):
    from swarm_alloc.perception.dbscan_inc import GroupTracker
    from swarm_alloc.perception.topsis import topsis
    from swarm_alloc.simulation.state import State

    st = State(t=0.0, enemies={u.id: u for u in s.waves[0].units}, groups=[],
               pool={fu.id: fu for fu in s.friendly})
    tr = GroupTracker(scfg["perception"]["eps_m"], scfg["perception"]["min_samples"])
    ids = st.enemy_ids
    tr.update(ids, np.vstack([st.enemies[i].pos for i in ids]))
    st.groups = topsis(list(tr.groups.values()), st.enemies)
    return st


def test_myopic_opt_respects_c1_and_c2():
    """近视最优：单架单目标（C1）+ 打则必达配额（C2）。"""
    from swarm_alloc.algorithms.myopic_opt import MyopicOpt

    st = _state_at_wave0(SCENES[0])
    X = MyopicOpt().solve(st, model)
    assert (X.sum(axis=1) <= 1).all(), "违反 C1"
    quotas = np.array([model.quota(p) for p in st.enemy_matrix()[1]])
    col = X.sum(axis=0)
    for c, q in zip(col, quotas):
        assert c == 0 or c >= q, "违反 C2"
    assert X.sum() <= len(st.pool_ids)


def test_myopic_opt_is_myopic():
    """它会用尽当前兵力（缺乏预见）——这正是"近视"的定义。"""
    from swarm_alloc.algorithms.myopic_opt import MyopicOpt

    st = _state_at_wave0(SCENES[0])
    X = MyopicOpt().solve(st, model)
    used = int(X.sum())
    # 波次目标总量远小于编制，若不近视不应把编制打光
    assert used == len(st.pool_ids)
