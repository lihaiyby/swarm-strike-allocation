"""M1 可复现性测试：同种子两次生成必须逐比特一致。"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm_alloc.config import load
from swarm_alloc.scenario.generator import gen_scene


def _fingerprint(scene) -> tuple:
    fp = [scene.seed, scene.R, scene.T, len(scene.friendly)]
    for w in scene.waves:
        for u in w.units:
            fp += [u.id, u.type_name, u.value, u.speed, u.p_kill, u.cluster]
            fp += list(np.round(u.pos, 10)) + list(np.round(u.vel, 10))
    for fu in scene.friendly:
        fp += [fu.id, fu.speed] + list(np.round(fu.pos, 10))
    return tuple(fp)


def test_same_seed_identical():
    scfg, etcfg, fcfg = load("scenario"), load("enemy_types"), load("friendly")
    for idx in (0, 25, 49):
        seed = scfg["seeds"][idx]
        a = gen_scene(seed, idx, scfg, etcfg["types"], fcfg)
        b = gen_scene(seed, idx, scfg, etcfg["types"], fcfg)
        assert _fingerprint(a) == _fingerprint(b), f"seed={seed} 两次生成不一致"


def test_r_rotation():
    scfg, etcfg, fcfg = load("scenario"), load("enemy_types"), load("friendly")
    Rs = [gen_scene(scfg["seeds"][i], i, scfg, etcfg["types"], fcfg).R for i in range(6)]
    assert Rs == [3000.0, 4000.0, 5000.0] * 2


def test_scene_counts_in_range():
    scfg, etcfg, fcfg = load("scenario"), load("enemy_types"), load("friendly")
    for idx in range(0, 50, 7):
        s = gen_scene(scfg["seeds"][idx], idx, scfg, etcfg["types"], fcfg)
        assert 100 <= s.n_enemy <= 150
        assert len(s.friendly) == 3 * s.n_enemy
        assert len(s.waves) == 3
