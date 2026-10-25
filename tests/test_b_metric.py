"""指标 b（威胁消除率）的口径测试。

覆盖任务 P0.5 的四项中的前三项：
① b 口径跨算法可比（分母与算法无关）
② 群划分是"划分"而非"覆盖"（不重不漏）
③ 一票否决语义正确
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm_alloc.algorithms import greedy  # noqa: F401
from swarm_alloc.algorithms.base import get
from swarm_alloc.config import ROOT, load
from swarm_alloc.evaluation.b_metric import group_elimination_rate
from swarm_alloc.model import Model
from swarm_alloc.simulation.simulate import simulate

scfg, etcfg = load("scenario"), load("enemy_types")
model = Model(load("eval"))

with open(ROOT / "data" / "scenes.pkl", "rb") as f:
    SCENES = pickle.load(f)


# ---------------------------------------------------------------- 基础公式

def test_all_dead_gives_one():
    P = {0: 0.95, 1: 0.91}
    b, num, den = group_elimination_rate(P, [[0, 1]], {0: 5, 1: 2})
    assert b == 1.0 and num == 7.0 and den == 7.0


def test_one_survivor_vetoes_whole_group():
    """一票否决：群内只剩 1 架未达阈值，整群价值不计入分子。"""
    P = {0: 0.95, 1: 0.91, 2: 0.87}      # 2 号未达 0.9
    b, num, den = group_elimination_rate(P, [[0, 1, 2]], {0: 5, 1: 2, 2: 2})
    assert num == 0.0 and den == 9.0 and b == 0.0


def test_missing_enemy_counts_as_not_dead():
    """未接战的敌机 P 缺失 → 按 0 处理，所在群不算歼灭。"""
    b, num, den = group_elimination_rate({0: 0.95}, [[0, 99]], {0: 5, 99: 2})
    assert b == 0.0 and den == 7.0


def test_value_weighting():
    """价值加权：大群全歼、小群未歼 → b = 大群价值占比。"""
    P = {0: 0.95, 1: 0.95, 2: 0.5}
    b, num, den = group_elimination_rate(
        P, [[0, 1], [2]], {0: 5, 1: 5, 2: 2})
    assert num == 10.0 and den == 12.0
    assert abs(b - 10.0 / 12.0) < 1e-12


def test_threshold_respected():
    P = {0: 0.9}
    b1, _, _ = group_elimination_rate(P, [[0]], {0: 1}, threshold=0.9)
    b2, _, _ = group_elimination_rate(P, [[0]], {0: 1}, threshold=0.91)
    assert b1 == 1.0 and b2 == 0.0      # 恰好等于阈值算歼灭（≥）


# ---------------------------------------------------------------- 真值团结构

@pytest.mark.parametrize("idx", [0, 17, 49])
def test_ground_truth_groups_partition(idx):
    """真值团必须是对全部敌机的**划分**：不重、不漏。"""
    s = SCENES[idx]
    groups = s.ground_truth_groups()
    flat = [eid for g in groups for eid in g]
    assert len(flat) == len(set(flat)) == s.n_enemy, "有重复或遗漏"
    assert set(flat) == {u.id for w in s.waves for u in w.units}


def test_ground_truth_is_noise_aware():
    """噪点必须各自独立成团（单元素群）。"""
    s = SCENES[0]
    noise_ids = {u.id for w in s.waves for u in w.units if u.is_noise}
    singleton = {g[0] for g in s.ground_truth_groups() if len(g) == 1}
    assert noise_ids <= singleton


def test_ground_truth_groups_deterministic():
    """同一场景两次调用结果一致（可复现）。"""
    s = SCENES[3]
    assert s.ground_truth_groups() == s.ground_truth_groups()


# ---------------------------------------------------------------- 跨算法可比性

@pytest.mark.parametrize("idx", [0, 25])
def test_b_denominator_is_algorithm_independent(idx):
    """核心性质：b 的分母 = 场景总价值，与算法无关。"""
    s = SCENES[idx]
    total = sum(u.value for w in s.waves for u in w.units)
    r_greedy = simulate(s, get("greedy"), model, scfg, etcfg)
    assert r_greedy.b_den == pytest.approx(total)
    assert r_greedy.b_threat == pytest.approx(r_greedy.b_num / r_greedy.b_den)


def test_b_denominator_stable_across_algorithms():
    """两个算法在同一场景上必须得到同一个分母（旧实现会漂移）。"""
    s = SCENES[0]
    r1 = simulate(s, get("greedy"), model, scfg, etcfg)
    r2 = simulate(s, get("ga_iap"), model, scfg, etcfg)
    assert r1.b_den == r2.b_den
    assert 0.0 <= r2.b_threat <= 1.0
