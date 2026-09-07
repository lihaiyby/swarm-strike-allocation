"""M2 冒烟实验：单场景 t=0 闭环（分组→威胁评估→算法分配→评估→可视化）。

用法：python experiments/exp0_smoke.py [seed] [--algo greedy]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import swarm_alloc.algorithms  # noqa: F401 自动注册全部算法
from swarm_alloc.algorithms.base import get
from swarm_alloc.config import ROOT, load
from swarm_alloc.evaluation.metrics import per_scene_metrics
from swarm_alloc.model import Model
from swarm_alloc.perception.dbscan_inc import GroupTracker
from swarm_alloc.perception.topsis import topsis
from swarm_alloc.simulation.state import State


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", nargs="?", type=int, default=1000)
    parser.add_argument("--algo", default="greedy", help="已注册的算法名")
    args = parser.parse_args()
    seed = args.seed
    scfg = load("scenario")
    model = Model(load("eval"))

    with open(ROOT / "data" / "scenes.pkl", "rb") as f:
        scenes = pickle.load(f)
    scene = next(s for s in scenes if s.seed == seed)

    # t=0 态势：第一波敌机 + 全部资源池
    wave0 = scene.waves[0]
    state = State(
        t=0.0,
        enemies={u.id: u for u in wave0.units},
        groups=[],
        pool={fu.id: fu for fu in scene.friendly},
    )

    # 感知：DBSCAN 分组 + TOPSIS
    tracker = GroupTracker(eps=scfg["perception"]["eps_m"],
                           min_samples=scfg["perception"]["min_samples"])
    eids = state.enemy_ids
    pos = np.vstack([state.enemies[i].pos for i in eids])
    tracker.update(eids, pos)
    state.groups = topsis(list(tracker.groups.values()), state.enemies)

    print(f"scene seed={seed} R={scene.R:.0f}m T={scene.T:.0f}s "
          f"敌机={len(eids)} 我方={len(state.pool)}")
    print(f"DBSCAN 分出 {len(state.groups)} 群（含噪点独立群）：")
    for g in sorted(state.groups, key=lambda g: -g.threat):
        print(f"  群{g.id:2d}  数量={len(g.member_ids):2d}  威胁度={g.threat:.3f}")

    # 算法求解（计时）
    algo = get(args.algo)
    t0 = time.perf_counter()
    X = algo.solve(state, model)
    elapsed = time.perf_counter() - t0

    # 约束检查
    values, p_kill, _ = state.enemy_matrix()
    quotas = np.array([model.quota(p, model.kill_threshold) for p in p_kill])
    group_of_enemy = {uid: g.id for g in state.groups for uid in g.member_ids}
    errs = model.check(X, state, quotas, group_of_enemy, {})
    print(f"\n约束检查: {'全部通过' if not errs else errs}")

    # 指标
    m = per_scene_metrics(X, state, model, elapsed)
    print(f"指标: a打击总收益={m['a_gain']:.1f}  b威胁消除率={m['b_threat']:.3f}  "
          f"d效费比={m['d_cost']:.3f}  出动={m['used']}架  耗时={m['time_s']*1000:.1f}ms")

    # 可视化：分组 + 分配结果
    out_dir = ROOT / "results" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot(state, X, scene, args.algo, out_dir / f"smoke_{args.algo}_seed{seed}.png")
    print(f"可视化 -> {out_dir/f'smoke_{args.algo}_seed{seed}.png'}")


def _plot(state: State, X: np.ndarray, scene, algo_name: str, out: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    epos = np.vstack([state.enemies[i].pos for i in state.enemy_ids])
    group_of = {uid: g.id for g in state.groups for uid in g.member_ids}
    gids = np.array([group_of[e] for e in state.enemy_ids])

    # 左：DBSCAN 分组结果（颜色=群）
    sc = ax1.scatter(epos[:, 0], epos[:, 1], c=gids, cmap="tab20", s=30)
    ax1.set_title(f"DBSCAN groups (eps=550m), threat-ranked: "
                  f"{[g.id for g in sorted(state.groups, key=lambda g:-g.threat)[:6]]}…")
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
    ax1.set_aspect("equal")

    # 右：分配结果（线=无人机→目标）
    fpos = state.pool_matrix()
    ax2.scatter(epos[:, 0], epos[:, 1], c="#C0504D", s=40, marker="^", label="enemy")
    ax2.scatter(fpos[:, 0], fpos[:, 1], c="#4C9F70", s=8, alpha=0.4, label="pool")
    ii, jj = np.nonzero(X)
    for i, j in zip(ii, jj):
        ax2.plot([fpos[i, 0], epos[j, 0]], [fpos[i, 1], epos[j, 1]],
                 "k-", lw=0.5, alpha=0.5)
    theta = np.linspace(0, np.pi, 200)
    ax2.plot(700 * np.cos(theta), 700 * np.sin(theta), "b--", lw=1.2)
    ax2.set_title(f"{algo_name} assignment: {int(X.sum())} sorties")
    ax2.set_xlabel("x (m)"); ax2.legend(fontsize=8)
    ax2.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
