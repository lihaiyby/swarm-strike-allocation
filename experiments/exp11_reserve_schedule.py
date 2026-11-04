"""exp11 · 逐波兵力预留调度（决策 D7-A 路线的核心实验）。

背景
----
`docs/算法区分度诊断.md` 证明：在本模型下，算法之间唯一的区分维度是
**每轮出动占可行容量的比例 α**。既有五个算法的 α 分别是：

| 算法 | α |
|---|---|
| greedy / cbba_ye / random_feasible | ≈ 1.00（每轮打满） |
| ga_iap | ≈ 0.69（主动预留） |
| myopic_opt | ≈ 2.59（过度投入） |

本实验把 α 从**常数**扩展为**逐波调度** $\alpha=(\alpha_1,\dots,\alpha_K)$，回答三个问题：

1. 逐波调度比最优常数 α 强多少？（调度的"增益空间"）
2. 最优调度长什么样？是"前紧后松"还是"前松后紧"？
3. 单一全局调度能否跨场景泛化？还是必须逐场景定制？

做法
----
在 greedy 外面套一层"本轮最多出动 α_k × 可行容量"的限额（按价值从低到高削），
对 α 做网格搜索。`greedy` 单场仅约 6 ms，因此可以直接穷举。

用法：
    python experiments/exp11_reserve_schedule.py --scenes 30 --tier normal
"""
from __future__ import annotations

import argparse
import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import swarm_alloc.algorithms  # noqa: F401,E402
from swarm_alloc.algorithms.base import AlgoBase, get  # noqa: E402
from swarm_alloc.config import ROOT, load  # noqa: E402
from swarm_alloc.scenario.generator import scene_file  # noqa: E402
from swarm_alloc.simulation.simulate import simulate  # noqa: E402

OUT = ROOT / "results" / "reserve_schedule"


class GreedySchedule(AlgoBase):
    """greedy + 逐波限额：第 k 轮最多出动 α_k × (本轮 greedy 原本要出动的架数)。"""

    name = "greedy_sched"

    def __init__(self, alphas):
        self.alphas = list(alphas)
        self._round = 0
        self._last_t = None

    def reset(self):
        self._round, self._last_t = 0, None

    def solve(self, state, model):
        t = state.t
        if self._last_t is None or t < self._last_t:      # 新战局
            self._round = 0
        elif t > self._last_t:                            # 下一触发点
            self._round += 1
        self._last_t = t

        X = get("greedy").solve(state, model)
        k = self._round
        alpha = self.alphas[min(k, len(self.alphas) - 1)]
        if alpha >= 0.999 or X.sum() == 0:
            return X

        cap = int(np.floor(X.sum() * alpha))
        values, _, _ = state.enemy_matrix()
        order = np.argsort(values)                        # 低价值优先削
        for j in order:
            if X.sum() <= cap:
                break
            rows = np.nonzero(X[:, j])[0]
            while len(rows) and X.sum() > cap:
                X[rows[-1], j] = 0
                rows = rows[:-1]
        return X


def run_one(scene, alphas, model, scfg, etcfg):
    algo = GreedySchedule(alphas)
    r = simulate(scene, algo, model, scfg, etcfg)
    return r.a_kill, r.b_threat, r.d_cost, r.n_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=30)
    ap.add_argument("--tier", default="normal")
    ap.add_argument("--grid", nargs="+", type=float,
                    default=[0.40, 0.55, 0.70, 0.85, 1.00])
    ap.add_argument("--depth", type=int, default=3,
                    help="调度长度（触发点数；不足则最后一档重复）")
    args = ap.parse_args()

    scfg, ecfg, etcfg = load("scenario"), load("eval"), load("enemy_types")
    from swarm_alloc.model import Model
    model = Model(ecfg)
    with open(ROOT / "data" / scene_file(args.tier, scfg), "rb") as f:
        scenes = pickle.load(f)[: args.scenes]
    OUT.mkdir(parents=True, exist_ok=True)

    grid = args.grid
    schedules = list(itertools.product(grid, repeat=args.depth))
    print(f"档位={args.tier}  场景={len(scenes)}  调度数={len(schedules)} "
          f"(α 取值 {grid}，深度 {args.depth})", flush=True)

    # ---- 1) 逐场景穷举 ----
    per_scene = []
    for s in scenes:
        best, best_sched = -1.0, None
        for sch in schedules:
            a = run_one(s, sch, model, scfg, etcfg)[0]
            if a > best:
                best, best_sched = a, sch
        per_scene.append(dict(seed=s.seed, a_kill_best=best,
                              sched="/".join(f"{x:.2f}" for x in best_sched)))
        print(f"  seed {s.seed}: 最优 {best:.1f}  α={best_sched}", flush=True)
    ps = pd.DataFrame(per_scene)
    ps.to_csv(OUT / "per_scene_best.csv", index=False)

    # ---- 2) 固定调度在全体场景上的均值（求全局最优）----
    print("\n评估各调度在全部场景上的均值…", flush=True)
    rows = []
    for sch in schedules:
        vals = [run_one(s, sch, model, scfg, etcfg)[0] for s in scenes]
        rows.append(dict(sched="/".join(f"{x:.2f}" for x in sch),
                         a_kill=float(np.mean(vals)),
                         a_kill_sd=float(np.std(vals, ddof=1))))
    g = pd.DataFrame(rows).sort_values("a_kill", ascending=False)
    g.to_csv(OUT / "global_schedule_rank.csv", index=False)

    print("\n" + "=" * 78)
    print("[1] 全局最优调度（单一策略跨场景）")
    print("=" * 78)
    print(g.head(10).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n[2] 常数 α 对照")
    const_rows = {}
    for a in grid:
        key = "/".join([f"{a:.2f}"] * args.depth)
        row = g[g.sched == key]
        if len(row):
            const_rows[a] = float(row.a_kill.iloc[0])
            print(f"  α≡{a:.2f}: {const_rows[a]:.2f}")
    best_const = max(const_rows.values()) if const_rows else float("nan")
    best_const_a = max(const_rows, key=const_rows.get) if const_rows else float("nan")
    best_global = float(g.a_kill.iloc[0])
    best_scene = float(ps.a_kill_best.mean())

    print(f"\n[3] 逐场景最优（调度上界，需每场定制）= {best_scene:.2f}")
    print(f"[4] 最优常数 α（α≡{best_const_a:.2f}）      = {best_const:.2f}")
    print(f"[5] 单一全局调度                        = {best_global:.2f}")
    print(f"\n调度增益（相对常数）= {(best_global - best_const) / best_const * 100:+.2f}%")
    print(f"每场定制的额外增益   = {(best_scene - best_global) / best_global * 100:+.2f}%")
    print(f"\n产出 -> {OUT}")


if __name__ == "__main__":
    main()
