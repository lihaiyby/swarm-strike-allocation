"""场景可视化预览：生成后肉眼检查分组可分性、分布合理性。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .scene import Scene

_TYPE_STYLE = {
    "recon":          {"c": "#4C9F70", "marker": "o", "label": "recon(v=2)"},
    "suicide_strike": {"c": "#E8A33D", "marker": "s", "label": "suicide(v=5)"},
    "fighter":        {"c": "#C0504D", "marker": "^", "label": "fighter(v=10)"},
    "area_damage":    {"c": "#7030A0", "marker": "D", "label": "area(v=10)"},
}


def preview_scene(scene: Scene, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13, 5.5))

    pos = np.vstack([u.pos for w in scene.waves for u in w.units])
    types = [u.type_name for w in scene.waves for u in w.units]
    waves_of = [wi for wi, w in enumerate(scene.waves) for _ in w.units]
    noise = [u.is_noise for w in scene.waves for u in w.units]

    # 左：3D 分布（颜色=波次）
    ax1 = fig.add_subplot(121, projection="3d")
    for wi, name in enumerate(["wave0 (t=0)", "wave1 (t=T/3)", "wave2 (t=2T/3)"]):
        m = [i for i, x in enumerate(waves_of) if x == wi]
        ax1.scatter(pos[m, 0], pos[m, 1], pos[m, 2], s=12, alpha=0.8, label=name)
    ax1.scatter([0], [0], [0], c="k", marker="*", s=150, label="base")
    ax1.set_title(f"Scene seed={scene.seed}  R={scene.R/1000:.0f}km  "
                  f"T={scene.T:.0f}s  n={scene.n_enemy}")
    ax1.legend(fontsize=8)

    # 右：俯视 x-y（颜色=类型，圈=噪点），画防线
    ax2 = fig.add_subplot(122)
    for tname, st in _TYPE_STYLE.items():
        m = [i for i, t in enumerate(types) if t == tname]
        if m:
            ax2.scatter(pos[m, 0], pos[m, 1], s=25, c=st["c"], marker=st["marker"],
                        alpha=0.85, label=st["label"])
    nm = [i for i, x in enumerate(noise) if x]
    if nm:
        ax2.scatter(pos[nm, 0], pos[nm, 1], s=90, facecolors="none",
                    edgecolors="gray", linewidths=1.2, label="noise")
    theta = np.linspace(0, np.pi, 200)  # 上半圆防线
    ax2.plot(700 * np.cos(theta), 700 * np.sin(theta), "b--", lw=1.2, label="defense 700m")
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("y (m)")
    ax2.set_title("top-down: enemy types & defense circle")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_aspect("equal")

    out = out_dir / f"scene_seed{scene.seed}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def preview_summary(scenes: list, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ns = [s.n_enemy for s in scenes]
    noise = [sum(1 for w in s.waves for u in w.units if u.is_noise) for s in scenes]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(ns, bins=12, color="#4C9F70", edgecolor="k")
    axes[0].set_title("enemy count n per scene"); axes[0].set_xlabel("n")
    axes[1].hist(noise, bins=10, color="#E8A33D", edgecolor="k")
    axes[1].set_title("noise points per scene"); axes[1].set_xlabel("count")
    out = out_dir / "summary.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out
