"""场景生成器主入口：按种子表生成 50 个场景 → data/scenes.pkl + 预览图。

用法：
    python -m swarm_alloc.scenario.generator                    # 生成默认档（easy 3n）→ scenes.pkl
    python -m swarm_alloc.scenario.generator --all-tiers        # 额外生成 normal/hard 档
    python -m swarm_alloc.scenario.generator --tiers normal hard

档位定义见 config/scenario.yaml 的 `tiers`（P2.1，决策 D1）。
**未指定档位时行为与历史完全一致**：`gen_scene(..., ratio=None)` 走 `friendly.count_ratio`。
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np

from ..config import ROOT, load
from .scene import FriendlyUnit, Scene
from .wave import gen_wave, sample_hemisphere


def _split_waves(n: int, count: int) -> list:
    base = [n // count] * count
    base[0] += n - sum(base)   # 余数归入第一波
    return base


def tier_ratios(scfg: dict) -> dict:
    """读取 config 里的档位定义 → {tier_name: ratio}。"""
    return {str(k): float(v) for k, v in (scfg.get("tiers") or {}).items()}


def scene_file(tier: str | None, scfg: dict | None = None) -> str:
    """档位 → 场景库文件名。`None` 或默认档返回 `scenes.pkl`。"""
    if tier is None:
        return "scenes.pkl"
    if scfg is None:
        scfg = load("scenario")
    default = scfg.get("tier_default")
    if tier == default:
        return "scenes.pkl"
    return f"scenes_{tier}.pkl"


def gen_scene(seed: int, idx: int, scfg: dict, type_cfgs: list, fcfg: dict,
              ratio: float | None = None) -> Scene:
    """生成单个场景。

    ratio : 我方兵力倍率；None 时用 `fcfg["count_ratio"]`（历史行为，默认 3.0）。
            取 `round(ratio × n)`，与 exp5 反事实实验的口径一致。
    """
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

    # 我方：内球面均匀部署，总数 = ratio × n（ratio 缺省取 count_ratio）
    defense_r = scfg["battlefield"]["defense_radius_m"]
    if ratio is None:
        n_friendly = int(fcfg["count_ratio"]) * n
    else:
        n_friendly = max(1, int(round(float(ratio) * n)))
    fpos = sample_hemisphere(rng, defense_r, n_friendly)
    fspeeds = rng.uniform(*fcfg["speed_mps"], size=n_friendly)
    friendly = [FriendlyUnit(id=i, pos=fpos[i], speed=float(fspeeds[i]))
                for i in range(n_friendly)]

    return Scene(seed=seed, R=R, T=T, waves=waves, friendly=friendly)


def generate_all(ratio: float | None = None) -> list:
    scfg, etcfg, fcfg = load("scenario"), load("enemy_types"), load("friendly")
    return [gen_scene(seed, i, scfg, etcfg["types"], fcfg, ratio=ratio)
            for i, seed in enumerate(scfg["seeds"])]


def build_tier(tier: str, write: bool = True) -> list:
    """按档位生成整套场景并（可选）写入 data/scenes_<tier>.pkl。"""
    scfg = load("scenario")
    ratios = tier_ratios(scfg)
    if tier not in ratios:
        raise KeyError(f"未知档位 {tier!r}；config 里可选 {sorted(ratios)}")
    scenes = generate_all(ratio=ratios[tier])
    if write:
        data_dir = ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        path = data_dir / scene_file(tier, scfg)
        with open(path, "wb") as f:
            pickle.dump(scenes, f)
        print(f"[{tier}] ratio={ratios[tier]} → {path} "
              f"（{len(scenes)} 场景，兵力 {min(len(s.friendly) for s in scenes)}"
              f"~{max(len(s.friendly) for s in scenes)}）")
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=int, default=3, help="预览图的场景个数")
    ap.add_argument("--tiers", nargs="+", default=None,
                    help="额外生成的档位名（见 config/scenario.yaml → tiers）")
    ap.add_argument("--all-tiers", action="store_true", help="生成全部已定义档位")
    ap.add_argument("--no-default", action="store_true",
                    help="跳过默认档（不重写 data/scenes.pkl）")
    args = ap.parse_args()

    scfg = load("scenario")
    ratios = tier_ratios(scfg)

    # 1) 默认档（历史行为，写 scenes.pkl）
    scenes = None
    if not args.no_default:
        scenes = generate_all()
        data_dir = ROOT / "data"
        data_dir.mkdir(exist_ok=True)
        with open(data_dir / "scenes.pkl", "wb") as f:
            pickle.dump(scenes, f)

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

    # 2) 其他档位（P2.1）
    targets: list = []
    if args.all_tiers:
        targets = [t for t in ratios if t != scfg.get("tier_default")]
    elif args.tiers:
        targets = list(args.tiers)
    for tier in targets:
        build_tier(tier)


if __name__ == "__main__":
    main()
