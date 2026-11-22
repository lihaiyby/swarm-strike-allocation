"""exp15 · 时序口径下的 12 算法对比（V2-P2 首轮）。

与 `exp1_main` 的区别
---------------------
| | exp1_main | **exp15（本脚本）** |
|---|---|---|
| 仿真内核 | `simulate`（4 个固定时刻、下达即命中） | `simulate_timed`（事件驱动、飞行时间、竞速判据） |
| 结束条件 | 到 $T$ 停 | **敌机全部了结且无在途** |
| 时间指标 | 整场累计 / 单次平均 | 单次平均（决策次数更多，故更强实时性要求） |

用法::

    python experiments/exp15_timed_main.py --tier normal --scenes 10
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import swarm_alloc.algorithms  # noqa: F401,E402
from swarm_alloc.algorithms.base import available, get  # noqa: E402
from swarm_alloc.config import ROOT, load  # noqa: E402
from swarm_alloc.model import Model  # noqa: E402
from swarm_alloc.scenario.generator import scene_file  # noqa: E402
from swarm_alloc.simulation.des import simulate_timed  # noqa: E402

OUT = ROOT / "results" / "timed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="normal")
    ap.add_argument("--scenes", type=int, default=10)
    ap.add_argument("--algos", nargs="+", default=None)
    ap.add_argument("--end-mode", default="engagement",
                    choices=["engagement", "fixed_t"])
    args = ap.parse_args()

    scfg, ecfg, etcfg, fcfg = load("scenario"), load("eval"), load("enemy_types"), load("friendly")
    model = Model(ecfg)
    algo_names = args.algos or available()

    with open(ROOT / "data" / scene_file(args.tier, scfg), "rb") as f:
        scenes = pickle.load(f)[: args.scenes]

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in algo_names:
        algo = get(name)
        for s in scenes:
            r = simulate_timed(s, algo, model, scfg, etcfg, fcfg,
                               end_mode=args.end_mode)
            rows.append(dict(
                algo=name, seed=s.seed, R=s.R, tier=args.tier,
                a_kill=r.a_kill, a_gain=r.a_gain, b_threat=r.b_threat,
                d_cost=r.d_cost, a_net=r.a_net,
                n_enemy=r.n_enemy, destroyed=r.n_destroyed,
                leaked=r.n_leaked, lost=r.n_lost, used=r.n_used,
                n_decisions=r.n_decisions,
                strike_hit=r.n_strike_hit, strike_void=r.n_strike_void,
                strike_late=r.n_strike_late,
                time_mean_ms=r.time_mean * 1000.0,
                time_p95_ms=r.time_p95 * 1000.0,
                time_max_ms=r.time_max * 1000.0,
                time_total_s=r.time_s,
            ))
        print(f"[exp15] {name} 完成", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"timed_{args.tier}_raw.csv", index=False)

    g = df.groupby("algo").agg(
        a_kill=("a_kill", "mean"), b=("b_threat", "mean"),
        d=("d_cost", "mean"), used=("used", "mean"),
        destroyed=("destroyed", "mean"), leaked=("leaked", "mean"),
        lost=("lost", "mean"), dec=("n_decisions", "mean"),
        hit=("strike_hit", "mean"), void=("strike_void", "mean"),
        late=("strike_late", "mean"),
        t_mean_ms=("time_mean_ms", "mean"), t_max_ms=("time_max_ms", "max"),
    ).round(2)
    g["放弃率%"] = (g.late / (g.used + g.late) * 100).round(1)
    g["突防率%"] = (g.leaked / df.groupby("algo").n_enemy.mean() * 100).round(1)
    g = g.sort_values("a_kill", ascending=False)
    g.to_csv(OUT / f"timed_{args.tier}_summary.csv")

    print("\n" + "=" * 110)
    print(f"[1] 时序口径主表（{args.tier}，{args.scenes} 场景，end_mode={args.end_mode}）")
    print("=" * 110)
    print(g[["a_kill", "b", "d", "destroyed", "leaked", "突防率%",
             "used", "放弃率%", "dec", "t_mean_ms", "t_max_ms"]].to_string())

    # 综合分（方案 A，基线 greedy）
    try:
        from swarm_alloc.evaluation.score import score_relative
        sc = score_relative(df, baseline="greedy")
        print("\n" + "=" * 110)
        print("[2] 综合分（方案 A，基线 greedy = 1.000；e 用单次平均耗时，时限 1 s）")
        print("=" * 110)
        print(sc[["r_a", "r_b", "r_d", "r_e", "score", "rank"]].round(4).to_string())
        sc[["r_a", "r_b", "r_d", "r_e", "score", "rank"]].to_csv(
            OUT / f"timed_{args.tier}_score.csv")
    except Exception as e:                     # noqa: BLE001
        print("综合分计算失败:", e)

    print(f"\n产出 -> {OUT}")


if __name__ == "__main__":
    main()
