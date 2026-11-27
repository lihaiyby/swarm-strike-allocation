"""【V2-P2-3】在途可见性（State.inflight_*）的单元测试。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm_alloc.simulation.state import State  # noqa: E402


class _U:
    def __init__(self, uid, pos, value=1.0, p_kill=0.7):
        self.id = uid
        self.pos = np.asarray(pos, dtype=float)
        self.value = value
        self.p_kill = p_kill


class _Model:
    kill_threshold = 0.9

    @staticmethod
    def quota(p, thr):
        return 5 if p <= 0.4 else (2 if p <= 0.7 else 1)


def _state(inflight_count=None, inflight_p=None):
    st = State(t=0.0,
               enemies={1: _U(1, [100, 0, 0], p_kill=0.7),
                        2: _U(2, [200, 0, 0], p_kill=0.4),
                        3: _U(3, [300, 0, 0], p_kill=0.9)},
               groups=[], pool={0: _U(0, [0, 0, 0])})
    st.inflight_count = inflight_count or {}
    st.inflight_p = inflight_p or {}
    return st


def test_no_inflight_means_full_quota():
    st = _state()
    need = st.remaining_quota(_Model, lambda e: st.enemies[e].p_kill)
    assert list(need) == [2, 5, 1]          # p=0.7→2、p=0.4→5、p=0.9→1
    assert st.effective_p0().tolist() == [0.0, 0.0, 0.0]


def test_partial_inflight_reduces_need():
    st = _state(inflight_count={1: 1})
    need = st.remaining_quota(_Model, lambda e: st.enemies[e].p_kill)
    assert list(need) == [1, 5, 1]          # 目标 1 已在途 1 架，只需再派 1 架


def test_saturated_target_needs_zero():
    st = _state(inflight_count={1: 2, 2: 7})
    need = st.remaining_quota(_Model, lambda e: st.enemies[e].p_kill)
    assert list(need) == [0, 0, 1]          # 目标 1 已饱和；目标 2 在途超配额也不为负


def test_inflight_p_is_cumulative():
    """在途累积期望毁伤 = 1 - Π(1-p)，与结算口径一致。"""
    st = _state(inflight_count={1: 2},
                inflight_p={1: 1 - (1 - 0.5) * (1 - 0.5)})
    assert abs(st.inflight_p_of(1) - 0.75) < 1e-12
    assert st.inflight_p_of(2) == 0.0


def test_inflight_ids_sorted():
    st = _state(inflight_count={3: 1, 1: 2})
    assert st.inflight_ids == [1, 3]
