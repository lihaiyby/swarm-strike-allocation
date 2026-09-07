"""毁伤公式与配额表测试（统一模型文档v1 §2 §5）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm_alloc.config import load
from swarm_alloc.model import Model

model = Model(load("eval"))


def test_damage_formula():
    assert abs(model.damage(0.7, 1) - 0.7) < 1e-12
    assert abs(model.damage(0.7, 2) - 0.91) < 1e-12
    assert abs(model.damage(0.4, 5) - (1 - 0.6 ** 5)) < 1e-12
    assert model.damage(0.9, 0) == 0.0


def test_quota_table():
    # p = 0.4 / 0.7 / 0.9 → 配额 5 / 2 / 1 架（达到 0.9）
    assert model.quota(0.4) == 5
    assert model.quota(0.7) == 2
    assert model.quota(0.9) == 1
    # 配额必达阈值
    for p in (0.4, 0.7, 0.9):
        assert model.damage(p, model.quota(p)) >= 0.9
        assert model.damage(p, model.quota(p) - 1) < 0.9
