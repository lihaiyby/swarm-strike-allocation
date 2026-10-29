"""exp5 · D1 先导实验：编制档位反事实（不改任何 config）。

在运行期按比例收缩 `scene.friendly`，把同一批场景在不同兵力档位下跑一遍全部算法，
并用三套打分口径给出每档排名——目的是回答"编制收到多紧，擂台才开始区分算法"。

用法：
    python experiments/exp5_fleet_ratio.py --scenes 20 --ratios 3.0 2.5 2.0
"""
from __future__ import annotations

import argparse
import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import swarm_alloc.algorithms  # noqa: F401 自动注册全部算法
from swarm_alloc.algorithms.base import available, get
from swarm_alloc.config import ROOT, load
from swarm_alloc.evaluation.oracle import scene_upper_bound
from swarm_alloc.evaluation.score import (DEFAULT_WEIGHTS, score_relative)
from swarm_alloc.model import Model
from swarm_alloc.simulation.simulate import simulate

SHRINK_SALT = 7777          # 收缩抽样用的固定盐，保证可复现


def shrink_scene(scene, ratio: float):
    """把编制从 3n 收缩到 ratio*n（随机抽取、固定种子、保序）。"""
    target = int(round(ratio * scene.n_enemy))
    if target >= len(scene.friendly):
        return scene, len(scene.friendly)
    if target < 1:
        target = 1
    rng = np.random.default_rng(scene.seed + SHRINK_SALT)
    idx = np.sort(rng.choice(len(scene.friendly), size=target, replace=False))
    s2 = copy.copy(scene)
    s2.friendly = [scene.friendly[i] for i in idx]
    return s2, target


def run_ratio(ratio: float, scenes, algo_names, model, scfg, etcfg):
    rows = []
    for s in scenes:
        s2, n_pool = shrink_scene(s, ratio)
        ob = scene_upper_bound(s2, model)          # 上界依赖兵力，必须按档重算
        for name in algo_names:
            r = simulate(s2, get(name), model, scfg, etcfg)
            rows.append(dict(
                ratio=ratio, algo=name, seed=s.seed, R=s.R, n_pool0=n_pool,
                a_gain=r.a_gain, b_threat=r.b_threat, d_cost=r.d_cost,
                time_s=r.time_s, used=r.n_used, destroyed=r.n_destroyed,
                leaked=r.n_leaked, lost=r.n_lost, n_enemy=r.n_enemy,
                a_opt=ob.a_opt, b_opt=ob.b_opt, d_opt=ob.d_opt, used_opt=ob.used,
                gap_a=(ob.a_opt - r.a_gain) / ob.a_opt if ob.a_opt > 0 else 0.0,
                gap_b=ob.b_opt - r.b_threat))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["ratio", "algo"]).agg(
        a=("a_gain", "mean"), b=("b_threat", "mean"), d=("d_cost", "mean"),
        gap_a=("gap_a", "mean"), used=("used", "mean"),
        leaked=("leaked", "mean"), time=("time_s", "mean")).reset_index()
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--ratios", nargs="+", type=float, default=[3.0, 2.5, 2.0])
    ap.add_argument("--algos", nargs="+", default=None)
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    scfg, etcfg = load("scenario"), load("enemy_types")
    model = Model(load("eval"))
    with open(ROOT / "data" / "scenes.pkl", "rb") as f:
        scenes = pickle.load(f)[: args.scenes]
    algo_names = args.algos or available()

    out_dir = ROOT / "results" / "exp5"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for ratio in args.ratios:
        print(f"[run] 编制 {ratio}n × {len(scenes)} 场景 × {len(algo_names)} 算法 ...", flush=True)
        frames.append(run_ratio(ratio, scenes, algo_names, model, scfg, etcfg))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out_dir / "fleet_ratio_metrics.csv", index=False)

    summ = summarise(df)
    summ.to_csv(out_dir / "fleet_ratio_summary.csv", index=False)

    print("\n" + "=" * 78)
    print("各档位 · 指标均值")
    print("=" * 78)
    for ratio in args.ratios:
        sub = summ[summ.ratio == ratio].set_index("algo").drop(columns="ratio")
        print(f"\n--- 编制 {ratio}n ---")
        print(sub.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 78)
    print("各档位 · 方案 A 排名（基线 greedy，阈值 6s，默认权重）")
    print("=" * 78)
    for ratio in args.ratios:
        sub = df[df.ratio == ratio][["algo", "seed", "a_gain", "b_threat",
                                     "d_cost", "time_s"]]
        try:
            a = score_relative(sub, DEFAULT_WEIGHTS, "greedy", "threshold", 6.0)
        except ValueError as e:
            print(f"  编制 {ratio}n: 跳过（{e}）")
            continue
        print(f"\n--- 编制 {ratio}n ---")
        cols = ["r_a", "r_b", "r_d", "r_e", "score", "rank"]
        print(a[cols].to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 78)
    print("排名随档位的翻转")
    print("=" * 78)
    ranks = {}
    for ratio in args.ratios:
        sub = df[df.ratio == ratio][["algo", "seed", "a_gain", "b_threat",
                                     "d_cost", "time_s"]]
        try:
            a = score_relative(sub, DEFAULT_WEIGHTS, "greedy", "threshold", 6.0)
            ranks[f"{ratio}n"] = a["score"]
        except ValueError:
            continue
    if ranks:
        tab = pd.DataFrame(ranks)
        tab["Δ(最松→最紧)"] = tab.iloc[:, -1] - tab.iloc[:, 0]
        print(tab.sort_values(tab.columns[0], ascending=False)
              .to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 78)
    print("区分度：各档位下 a / b / d 的极差与最优性间隙")
    print("=" * 78)
    for ratio in args.ratios:
        sub = df[df.ratio == ratio]
        ra = sub.groupby("algo").a_gain.mean()
        rb = sub.groupby("algo").b_threat.mean()
        rd = sub.groupby("algo").d_cost.mean()
        ga_ = sub.groupby("algo").gap_a.mean()
        print(f"  编制 {ratio}n: a 极差={ra.max()-ra.min():7.1f} "
              f"({ra.max()/max(ra.min(),1e-9):5.2f}×) | "
              f"b 极差={rb.max()-rb.min():.3f} | "
              f"d 极差={rd.max()-rd.min():.3f} | "
              f"gap_a {ga_.min():.2%}~{ga_.max():.2%}")

    print(f"\n产出 -> {out_dir}")


if __name__ == "__main__":
    main()
