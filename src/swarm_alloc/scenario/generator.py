"""场景生成器主入口：按种子表生成 50 个场景 → data/scenes.pkl + 预览图。

用法：python -m swarm_alloc.scenario.generator [--preview N]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from ..config import ROOT, load
from .scene import FriendlyUnit, Scene
from .wave import gen_wave, sample_hemisphere


def _split_waves(n: int, count: int) -> list:
    base = [n // count] * count
    base[0] += n - sum(base)   # 余数归入第一波
    return base


def gen_scene(seed: int, idx: int, scfg: dict, type_cfgs: list, fcfg: dict) -> Scene:
    rng = np.random.default_rng(seed)
    R = float(scfg["battlefield"]["R_choices_m"][idx % len(scfg["battlefield"]["R_choices_m"])])
    T = R / scfg["battlefield"]["enemy_max_speed_mps"]

    n = int(rng.integers(scfg["enemy_count"]["n_min"], scfg["enemy_count"]["n_max"] + 1))
    n_waves = scfg["waves"]["count"]
    wave_ns = _split_waves(n, n_waves)

    waves, next_id = [], 0
    for wi, (frac, nw) in enumerate(zip(scfg["waves"]["times_frac"], wave_ns)):
        wave, next_id = gen_wave(rng, frac * T, nw, R, scfg, type_cfgs, next_id)
        waves.append(wave)

    # 我方：内球面均匀部署，总数 = count_ratio × n
    defense_r = scfg["battlefield"]["defense_radius_m"]
    n_friendly = int(fcfg["count_ratio"]) * n
    fpos = sample_hemisphere(rng, defense_r, n_friendly)
    fspeeds = rng.uniform(*fcfg["speed_mps"], size=n_friendly)
    friendly = [FriendlyUnit(id=i, pos=fpos[i], speed=float(fspeeds[i]))
                for i in range(n_friendly)]

    return Scene(seed=seed, R=R, T=T, waves=waves, friendly=friendly)


def generate_all() -> list:
    scfg, etcfg, fcfg = load("scenario"), load("enemy_types"), load("friendly")
    return [gen_scene(seed, i, scfg, etcfg["types"], fcfg)
            for i, seed in enumerate(scfg["seeds"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=int, default=3, help="预览图的场景个数")
    args = ap.parse_args()

    scenes = generate_all()
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    with open(data_dir / "scenes.pkl", "wb") as f:
        pickle.dump(scenes, f)

    # 汇总表
    rows = [(s.seed, s.R, s.n_enemy, len(s.friendly),
             sum(1 for w in s.waves for u in w.units if u.is_noise)) for s in scenes]
    import pandas as pd
    df = pd.DataFrame(rows, columns=["seed", "R_m", "n_enemy", "n_friendly", "n_noise"])
    res_dir = ROOT / "results"
    res_dir.mkdir(exist_ok=True)
    df.to_csv(res_dir / "scene_summary.csv", index=False)

    from .visualize import preview_scene, preview_summary
    for s in scenes[: args.preview]:
        preview_scene(s, res_dir / "scene_preview")
    preview_summary(scenes, res_dir / "scene_preview")

    print(f"生成 {len(scenes)} 个场景 -> {data_dir/'scenes.pkl'}")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
