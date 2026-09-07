"""eps 标定：对样本场景的第一波敌机画 k-distance 图，拐点即建议 eps。

用法：python -m swarm_alloc.perception.calibrate_eps [--min-samples 3] [--scenes 10]
结果写入 results/eps_calibration.png，并打印建议 eps（需人工确认后填入 config/scenario.yaml）。
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..config import ROOT, load
from ..scenario.generator import gen_scene


def kdist_curve(pos: np.ndarray, k: int) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(pos)
    dist, _ = nn.kneighbors(pos)
    return np.sort(dist[:, k])


def knee_point(y: np.ndarray) -> int:
    """最大曲率近似：点到首末连线距离最大处。"""
    x = np.arange(len(y), dtype=float)
    dx, dy = len(y) - 1.0, y[-1] - y[0]
    d = np.abs(dx * (y - y[0]) - dy * (x - 0.0)) / np.hypot(dx, dy)
    return int(np.argmax(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-samples", type=int, default=3)
    ap.add_argument("--scenes", type=int, default=10)
    args = ap.parse_args()

    scfg, etcfg = load("scenario"), load("enemy_types")
    fig, ax = plt.subplots(figsize=(8, 5))
    suggestions = []
    for idx in range(args.scenes):
        scene = gen_scene(scfg["seeds"][idx], idx, scfg, etcfg["types"], load("friendly"))
        pos = np.vstack([u.pos for u in scene.waves[0].units])
        y = kdist_curve(pos, args.min_samples)
        ki = knee_point(y)
        suggestions.append(y[ki])
        ax.plot(y, alpha=0.6, label=f"R={scene.R/1000:.0f}km knee≈{y[ki]:.0f}m")
    ax.set_xlabel("points sorted by distance")
    ax.set_ylabel(f"{args.min_samples}-NN distance (m)")
    ax.set_title(f"k-distance curves (first wave, {args.scenes} scenes)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = ROOT / "results" / "eps_calibration.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"各场景建议 eps: {[round(s) for s in suggestions]}")
    print(f"建议 eps 中位数: {float(np.median(suggestions)):.0f} m -> 图 {out}")


if __name__ == "__main__":
    main()
