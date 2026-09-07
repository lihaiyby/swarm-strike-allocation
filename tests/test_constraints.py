"""约束检查与可行性恢复测试。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm_alloc.config import load
from swarm_alloc.model import Model
from swarm_alloc.perception.dbscan_inc import Group
from swarm_alloc.simulation.state import State
from swarm_alloc.scenario.scene import EnemyUnit, FriendlyUnit

model = Model(load("eval"))


def _toy_state():
    enemies = {
        0: EnemyUnit(0, np.array([3000., 0, 0]), np.zeros(3), 50, "suicide_strike", 5, 5, 0.7, False, 0),
        1: EnemyUnit(1, np.array([3100., 0, 0]), np.zeros(3), 50, "recon", 2, 10, 0.4, False, 0),
    }
    pool = {i: FriendlyUnit(i, np.array([0., 700., 0]), 55.0) for i in range(10)}
    groups = [Group(id=0, member_ids=[0, 1], threat=0.8)]
    return State(t=0.0, enemies=enemies, groups=groups, pool=pool)


def test_c1_single_target_per_drone():
    s = _toy_state()
    X = np.zeros((10, 2), dtype=int)
    X[0, 0] = X[0, 1] = 1                      # 同一架打两个目标 → 违反 C1
    errs = model.check(X, s, np.array([2, 5]), {0: 0, 1: 0}, {})
    assert any("C1" in e for e in errs)


def test_c2_quota_lower_bound():
    s = _toy_state()
    X = np.zeros((10, 2), dtype=int)
    X[0, 1] = 1                                # 目标1需5架(p=0.4)，只派1架 → 违反 C2
    errs = model.check(X, s, np.array([2, 5]), {0: 0, 1: 0}, {})
    assert any("C2" in e for e in errs)


def test_c5_frozen_not_assignable():
    s = _toy_state()
    s.frozen_ids = [3]
    X = np.zeros((10, 2), dtype=int)
    X[3, 0] = 1; X[4, 0] = 1                   # 3号已冻结仍被派出 → 违反 C5
    errs = model.check(X, s, np.array([2, 5]), {0: 0, 1: 0}, {})
    assert any("C5" in e for e in errs)


def test_feasible_drop_low_threat():
    g_high = Group(id=0, member_ids=[0], threat=0.9)
    g_low = Group(id=1, member_ids=[1], threat=0.1)
    demand = {0: 8, 1: 5}
    kept = model.select_groups([g_high, g_low], demand, pool_size=10)
    assert [g.id for g in kept] == [0]         # 资源不够，放弃低威胁群


def test_greedy_smoke_no_violation():
    """贪心在真实场景 t=0 无约束违规且达到配额。"""
    import pickle
    from swarm_alloc.config import ROOT
    from swarm_alloc.algorithms.base import get
    from swarm_alloc.algorithms import greedy  # noqa: F401
    from swarm_alloc.perception.dbscan_inc import GroupTracker
    from swarm_alloc.perception.topsis import topsis

    with open(ROOT / "data" / "scenes.pkl", "rb") as f:
        scenes = pickle.load(f)
    scene = scenes[0]
    state = State(t=0.0,
                  enemies={u.id: u for u in scene.waves[0].units},
                  groups=[], pool={fu.id: fu for fu in scene.friendly})
    tracker = GroupTracker(550, 3)
    tracker.update(state.enemy_ids,
                   np.vstack([state.enemies[i].pos for i in state.enemy_ids]))
    state.groups = topsis(list(tracker.groups.values()), state.enemies)

    X = get("greedy").solve(state, model)
    quotas = np.array([model.quota(p) for p in
                       [state.enemies[e].p_kill for e in state.enemy_ids]])
    col = X.sum(axis=0)
    for j, c in enumerate(col):
        assert c == 0 or c >= quotas[j]        # 打则必达配额
    assert (X.sum(axis=1) <= 1).all()          # 单架单目标
