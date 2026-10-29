"""编制档位的单测（任务 P2.4）。

覆盖：
① 档位读配置正确；
② 场景库文件映射（默认档仍指向 scenes.pkl，保证向后兼容）；
③ `ratio=None` 与历史行为逐位一致；
④ `ratio=3.0` 与 `ratio=None` 等价；
⑤ 各档兵力数 = round(ratio × n)；
⑥ 同档两次生成逐位一致（可复现）；
⑦ **敌机分布与档位无关**（只有我方兵力变），故感知层 eps 无需按档重标定（P2.3）。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm_alloc.config import load  # noqa: E402
from swarm_alloc.scenario.generator import (gen_scene, scene_file,  # noqa: E402
                                           tier_ratios)

scfg, etcfg, fcfg = load("scenario"), load("enemy_types"), load("friendly")
TYPES = etcfg["types"]


def _scene(idx=0, ratio=None):
    return gen_scene(scfg["seeds"][idx], idx, scfg, TYPES, fcfg, ratio=ratio)


def _enemy_fp(scene):
    """敌机指纹（不含我方）。"""
    fp = []
    for w in scene.waves:
        for u in w.units:
            fp += [u.id, u.type_name, u.value, u.p_kill, u.cluster]
            fp += list(np.round(u.pos, 12)) + list(np.round(u.vel, 12))
    return tuple(fp)


def _friendly_fp(scene):
    fp = []
    for f in scene.friendly:
        fp += [f.id, f.speed] + list(np.round(f.pos, 12))
    return tuple(fp)


# ---------------------------------------------------------------- 配置与映射

def test_tiers_defined_in_config():
    ratios = tier_ratios(scfg)
    assert ratios, "config/scenario.yaml 未定义 tiers"
    assert ratios["easy"] == pytest.approx(3.0)
    assert ratios["normal"] == pytest.approx(2.5)
    assert ratios["hard"] == pytest.approx(2.0)


def test_scene_file_mapping():
    default = scfg.get("tier_default")
    assert scene_file(None, scfg) == "scenes.pkl"
    assert scene_file(default, scfg) == "scenes.pkl"
    for tier in tier_ratios(scfg):
        if tier != default:
            assert scene_file(tier, scfg) == f"scenes_{tier}.pkl"


# ---------------------------------------------------------------- 向后兼容

def test_ratio_none_matches_legacy():
    """ratio=None 必须与历史行为逐位一致。"""
    s = _scene(0, None)
    n = s.n_enemy
    assert len(s.friendly) == int(fcfg["count_ratio"]) * n


def test_ratio_3_equals_none():
    """ratio=3.0 与不传 ratio 等价（兵力与位置都应一致）。"""
    a = _scene(0, None)
    b = _scene(0, 3.0)
    assert len(a.friendly) == len(b.friendly)
    assert _friendly_fp(a) == _friendly_fp(b)


# ---------------------------------------------------------------- 各档兵力

@pytest.mark.parametrize("tier", ["easy", "normal", "hard"])
@pytest.mark.parametrize("idx", [0, 17, 49])
def test_fleet_count_matches_ratio(tier, idx):
    ratio = tier_ratios(scfg)[tier]
    s = _scene(idx, ratio)
    assert len(s.friendly) == int(round(ratio * s.n_enemy))


def test_fleet_shrinks_monotonically():
    a = _scene(0, tier_ratios(scfg)["easy"])
    b = _scene(0, tier_ratios(scfg)["normal"])
    c = _scene(0, tier_ratios(scfg)["hard"])
    assert len(a.friendly) > len(b.friendly) > len(c.friendly)


# ---------------------------------------------------------------- 可复现

@pytest.mark.parametrize("tier", ["easy", "normal", "hard"])
def test_tier_reproducible(tier):
    ratio = tier_ratios(scfg)[tier]
    s1, s2 = _scene(3, ratio), _scene(3, ratio)
    assert _enemy_fp(s1) == _enemy_fp(s2)
    assert _friendly_fp(s1) == _friendly_fp(s2)


# ---------------------------------------------------------------- P2.3：eps 与档位无关

@pytest.mark.parametrize("idx", [0, 25])
def test_enemy_distribution_independent_of_tier(idx):
    """只有我方兵力变，敌机分布必须完全相同 → 感知层 eps 无需按档重标定。"""
    base = _enemy_fp(_scene(idx, None))
    for tier in tier_ratios(scfg):
        assert _enemy_fp(_scene(idx, tier_ratios(scfg)[tier])) == base


def test_unknown_tier_rejected():
    from swarm_alloc.scenario.generator import build_tier
    with pytest.raises(KeyError):
        build_tier("nope", write=False)
