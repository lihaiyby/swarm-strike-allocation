"""exp1 主实验：全部已注册算法 × 50 场景 → 指标表 + 全套可视化。

用法：python experiments/exp1_main.py [--algos greedy] [--scenes 50]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import swarm_alloc.algorithms  # noqa: F401 自动注册全部算法
from swarm_alloc.algorithms.base import available, get
from swarm_alloc.config import ROOT, load
from swarm_alloc.model import Model
from swarm_alloc.simulation.simulate import simulate


def run_all(algo_names: list, n_scenes: int) -> pd.DataFrame:
    scfg, etcfg = load("scenario"), load("enemy_types")
    model = Model(load("eval"))
    with open(ROOT / "data" / "scenes.pkl", "rb") as f:
        scenes = pickle.load(f)[:n_scenes]

    rows, details = [], {}
    for name in algo_names:
        for s in scenes:
            algo = get(name)
            r = simulate(s, algo, model, scfg, etcfg)
            rows.append(dict(algo=name, seed=s.seed, R=s.R, a_gain=r.a_gain,
                             b_threat=r.b_threat, d_cost=r.d_cost, time_s=r.time_s,
                             n_enemy=r.n_enemy, destroyed=r.n_destroyed,
                             leaked=r.n_leaked, lost=r.n_lost, used=r.n_used))
            details[(name, s.seed)] = r
    return pd.DataFrame(rows), details


def plot_metrics(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col, title in [
        (axes[0, 0], "a_gain", "a: total expected value destroyed"),
        (axes[0, 1], "b_threat", "b: threat elimination rate"),
        (axes[1, 0], "d_cost", "d: value per sortie"),
        (axes[1, 1], "time_s", "e: total solve time per scene (s)"),
    ]:
        for name, grp in df.groupby("algo"):
            ax.hist(grp[col], bins=15, alpha=0.7, label=name, edgecolor="k")
        ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("exp1: metric distributions over 50 scenes")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_timeline(r, scene, out: Path):
    h = pd.DataFrame(r.history)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(h.t, h.n_field, "r-o", label="enemies in field")
    ax1.plot(h.t, h.n_destroyed, "g--s", label="destroyed (cum)")
    ax1.plot(h.t, h.n_leaked, "k-.", label="leaked (cum)")
    ax1.set_xlabel("t (s)"); ax1.set_ylabel("count")
    ax2 = ax1.twinx()
    ax2.plot(h.t, h.n_pool, "b-", alpha=0.5, label="pool left")
    ax2.set_ylabel("pool", color="b")
    for frac, name in [(0, "wave0"), (1 / 3, "wave1"), (2 / 3, "wave2")]:
        ax1.axvline(frac * scene.T, color="gray", ls=":", lw=1)
        ax1.text(frac * scene.T, ax1.get_ylim()[1] * 0.95, name, fontsize=8, color="gray")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")
    ax1.set_title(f"battle timeline: seed={scene.seed} algo={r.algo} R={scene.R/1000:.0f}km")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_snapshots(r, scene, out: Path):
    n = len(r.snapshots)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (t, epos, X, ppos, engaged) in zip(axes, r.snapshots):
        if len(ppos):
            ax.scatter(ppos[:, 0], ppos[:, 1], c="#4C9F70", s=6, alpha=0.4, label="pool")
        if len(epos):
            ax.scatter(epos[:, 0], epos[:, 1], c="#C0504D", s=45, marker="^", label="enemy")
        # 分配线需要 pool 全量坐标——快照中 pool 是分配后的，改用 engaged 敌机坐标画接战点
        for p in engaged:
            ax.scatter([p[0]], [p[1]], c="orange", s=90, marker="x", linewidths=2)
        theta = np.linspace(0, np.pi, 200)
        ax.plot(700 * np.cos(theta), 700 * np.sin(theta), "b--", lw=1)
        ax.set_title(f"t={t:.0f}s: engaged {len(engaged)}, field {len(epos)}")
        ax.set_xlabel("x (m)"); ax.set_aspect("equal")
    axes[0].set_ylabel("y (m)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"wave snapshots: seed={scene.seed} algo={r.algo}  (x = engaged this round)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", nargs="+", default=None)
    ap.add_argument("--scenes", type=int, default=50)
    ap.add_argument("--detail-seed", type=int, default=1000)
    args = ap.parse_args()
    algo_names = args.algos or available()

    df, details = run_all(algo_names, args.scenes)
    out_dir = ROOT / "results" / "exp1"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "exp1_metrics.csv", index=False)

    summary = df.groupby("algo")[["a_gain", "b_threat", "d_cost", "time_s",
                                  "destroyed", "leaked", "lost", "used"]].agg(["mean", "std"])
    summary.to_csv(out_dir / "exp1_summary.csv")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    plot_metrics(df, out_dir / "metrics_dist.png")

    scfg = load("scenario")
    with open(ROOT / "data" / "scenes.pkl", "rb") as f:
        scenes = pickle.load(f)
    scene = next(s for s in scenes if s.seed == args.detail_seed)
    r = details[(algo_names[0], args.detail_seed)]
    plot_timeline(r, scene, out_dir / f"timeline_seed{scene.seed}.png")
    plot_snapshots(r, scene, out_dir / f"snapshots_seed{scene.seed}.png")
    print(f"图表 -> {out_dir}")


if __name__ == "__main__":
    main()
