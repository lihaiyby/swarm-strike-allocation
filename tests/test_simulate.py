"""仿真主循环测试：可复现、账目守恒、指标合法。"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from swarm_alloc.algorithms.base import get
from swarm_alloc.algorithms import greedy  # noqa: F401
from swarm_alloc.config import ROOT, load
from swarm_alloc.model import Model
from swarm_alloc.simulation.simulate import simulate

scfg, etcfg = load("scenario"), load("enemy_types")
model = Model(load("eval"))

with open(ROOT / "data" / "scenes.pkl", "rb") as f:
    SCENES = pickle.load(f)


@pytest.mark.parametrize("idx", [0, 25, 49])
def test_simulate_reproducible(idx):
    s = SCENES[idx]
    r1 = simulate(s, get("greedy"), model, scfg, etcfg)
    r2 = simulate(s, get("greedy"), model, scfg, etcfg)
    assert (r1.a_gain, r1.b_threat, r1.n_destroyed, r1.n_leaked, r1.n_lost, r1.n_used) == \
           (r2.a_gain, r2.b_threat, r2.n_destroyed, r2.n_leaked, r2.n_lost, r2.n_used)


def test_accounting_conserved():
    s = SCENES[0]
    r = simulate(s, get("greedy"), model, scfg, etcfg)
    # 敌机账目：被摧毁 + 漏网 ≤ 总数（剩余在空的在最后结算时仍存活）
    assert r.n_destroyed + r.n_leaked <= r.n_enemy
    # 我方账目：出动 + 战损 ≤ 初始资源池
    assert r.n_used + r.n_lost <= 3 * r.n_enemy
    assert 0.0 <= r.b_threat <= 1.0
    assert r.a_gain >= 0 and r.d_cost >= 0


def test_pool_never_negative():
    s = SCENES[10]
    r = simulate(s, get("greedy"), model, scfg, etcfg)
    assert all(h["n_pool"] >= 0 for h in r.history)
    assert all(h["n_field"] >= 0 for h in r.history)
