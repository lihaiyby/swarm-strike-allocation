"""【V2-P6】紧迫性加权与威胁特征配置的单元测试。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm_alloc.perception.topsis import (FEATURES, _radial,  # noqa: E402
                                           features_from_config)
from swarm_alloc.simulation.state import State  # noqa: E402


class _U:
    def __init__(self, uid, pos, vel, value=5.0, p_kill=0.7, speed=50.0):
        self.id = uid
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)
        self.value = value
        self.p_kill = p_kill
        self.speed = speed


def _state(beta=0.0):
    # 目标 1 径向接近 50 m/s；目标 2 侧向（径向分量 0）；目标 3 远离
    st = State(t=0.0,
               enemies={1: _U(1, [1000, 0, 0], [-50, 0, 0]),
                        2: _U(2, [1000, 0, 0], [0, 50, 0]),
                        3: _U(3, [1000, 0, 0], [50, 0, 0])},
               groups=[], pool={})
    st.urgency_beta = beta
    return st


def test_beta_zero_keeps_values():
    st = _state(0.0)
    vals, _, _ = st.enemy_matrix()
    assert vals.tolist() == [5.0, 5.0, 5.0]


def test_urgency_ranks_by_closing_rate():
    st = _state(0.0)
    u = st._urgency()
    assert abs(u[0] - 1.0) < 1e-9        # 迎面最快 → 最紧迫
    assert abs(u[1] - 0.5) < 1e-9        # 侧向 → 居中
    assert abs(u[2] - 0.0) < 1e-9        # 远离 → 最不紧迫


def test_beta_amplifies_urgent_targets():
    st = _state(1.0)
    vals, _, _ = st.enemy_matrix()
    assert vals[0] > vals[1] > vals[2]   # 紧迫的目标在算法眼里更值钱
    assert abs(vals[0] - 10.0) < 1e-9    # 5·(1+1·1)


def test_all_same_direction_is_neutral():
    st = State(t=0.0,
               enemies={1: _U(1, [1000, 0, 0], [-50, 0, 0]),
                        2: _U(2, [2000, 0, 0], [-50, 0, 0])},
               groups=[], pool={})
    st.urgency_beta = 1.0
    assert st._urgency().tolist() == [0.5, 0.5]
    vals, _, _ = st.enemy_matrix()
    assert abs(vals[0] - 5.0 * 1.5) < 1e-9   # 全体同向时价值同乘 1.5，排序不变


def test_radial_closure_and_eta():
    class _P:
        pos = np.array([1000.0, 0, 0])
        vel = np.array([-50.0, 0, 0])
    clos, eta = _radial([_P()], 700.0)
    assert abs(clos - 50.0) < 1e-9
    assert abs(eta - (1000 - 700) / 50.0) < 1e-9


def test_features_from_config_defaults_to_old_set():
    feats, r = features_from_config({})
    assert feats == ["value", "speed", "dist"]
    assert r == 700.0


def test_features_from_config_reads_yaml():
    scfg = {"perception": {"threat": {"features": ["value", "closure"],
                                      "defense_radius_m": 650.0}},
            "battlefield": {"defense_radius_m": 700.0}}
    feats, r = features_from_config(scfg)
    assert feats == ["value", "closure"]
    assert r == 650.0
