"""exp12 · 【批注 B1/B3 落地】分组方式 × 距离方式 的完整对照实验。

背景
----
`进度汇报1105.docx` 要求：
- **三种我方分组方式都测**（even / sector / nearest），我方分组 + 组对组对抗；
- **四种加入距离的方式都测**（A 毁伤衰减 / B 硬射程 / C 代价项 / D 时间延迟）。

本脚本覆盖 **B3 的三种分组** 与 **B1 的 A/B/C 三种距离**（D 时间延迟另见 exp13）；
每种配置都在同一批场景、同一批算法上跑，输出长表 + 汇总 + 排名。

用法
----
    python experiments/exp12_annotations.py --kind grouping --scenes 15
    python experiments/exp12_annotations.py --kind distance --scenes 15
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
from swarm_alloc.simulation.simulate import simulate  # noqa: E402

OUT = ROOT / "results" / "annotations"

GROUPING = {
    "none": None,
    "even": {"enabled": True, "mode": "even"},
    "sector": {"enabled": True, "mode": "sector"},
    "nearest": {"enabled": True, "mode": "nearest"},
}

DISTANCE = {
    "none": ({}, {}),
    "A_decay": ({"enabled": True, "lambda": 2.0, "ref_range_m": 3000.0,
                 "hard_frac": 0.0}, {}),
    "B_hard": ({"enabled": True, "lambda": 2.0, "ref_range_m": 3000.0,
                "hard_frac": 0.5}, {}),
    "C_cost": ({}, {"enabled": True, "c": 0.0001}),
}


def cfg_for(kind: str, name: str) -> dict:
    """构造该配置的 eval 配置。"""
    base = load("eval")
    if kind == "grouping":
        gp = GROUPING[name]
        return {**base, "grouping": gp} if gp else dict(base)
    rd, dc = DISTANCE[name]
    out = dict(base)
    if rd:
        out["range_decay"] = {"enabled": True, "lambda": rd["lambda"],
                              "ref_range_m": rd["ref_range_m"],
                              "hard_frac": rd["hard_frac"]}
    if dc:
        out["distance_cost"] = dc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["grouping", "distance"], required=True)
    ap.add_argument("--tiers", nargs="+", default=["normal", "hard"])
    ap.add_argument("--scenes", type=int, default=15)
    ap.add_argument("--algos", nargs="+", default=None)
    args = ap.parse_args()

    scfg, etcfg = load("scenario"), load("enemy_types")
    algo_names = args.algos or available()
    names = list(GROUPING) if args.kind == "grouping" else list(DISTANCE)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for tier in args.tiers:
        with open(ROOT / "data" / scene_file(tier, scfg), "rb") as f:
            scenes = pickle.load(f)[: args.scenes]
        for name in names:
            model = Model(cfg_for(args.kind, name))
            for s in scenes:
                for algo in algo_names:
                    r = simulate(s, get(algo), model, scfg, etcfg)
                    rows.append(dict(
                        tier=tier, config=name, algo=algo, seed=s.seed,
                        a_kill=r.a_kill, b=r.b_threat, d=r.d_cost,
                        a_gain=r.a_gain, a_net=r.a_net, sum_dist=r.sum_dist,
                        used=r.n_used, lost=r.n_lost, leaked=r.n_leaked,
                        c3_cross=r.c3_cross, c2_dropped=r.c2_dropped,
                        time_s=r.time_s))
            print(f"[{args.kind}] tier={tier} config={name} 完成", flush=True)

    df = pd.DataFrame(rows)
    tag = args.kind
    df.to_csv(OUT / f"{tag}_raw.csv", index=False)

    summ = df.groupby(["tier", "config", "algo"]).agg(
        a_kill=("a_kill", "mean"), b=("b", "mean"), d=("d", "mean"),
        a_net=("a_net", "mean"), used=("used", "mean"),
        cross=("c3_cross", "mean"), dropped=("c2_dropped", "mean"),
        t=("time_s", "mean")).round(3)
    summ.to_csv(OUT / f"{tag}_summary.csv")

    print("\n" + "=" * 90)
    print("[1] a_kill 均值（行=配置，列=算法）")
    print("=" * 90)
    for tier in args.tiers:
        p = df[df.tier == tier].pivot_table(index="config", columns="algo",
                                            values="a_kill")
        print(f"\n--- {tier} ---")
        print(p.round(1).to_string())

    print("\n" + "=" * 90)
    print("[2] 名次（每档每配置内按 a_kill 排序，1 最好）")
    print("=" * 90)
    for tier in args.tiers:
        p = df[df.tier == tier].pivot_table(index="config", columns="algo",
                                            values="a_kill")
        print(f"\n--- {tier} ---")
        print(p.rank(axis=1, ascending=False).astype(int).to_string())

    print("\n" + "=" * 90)
    print(f"[3] 区分度：各配置下 a_kill 的极差（越大越能区分算法）")
    print("=" * 90)
    for tier in args.tiers:
        sub = df[df.tier == tier]
        sp = sub.groupby(["config", "algo"]).a_kill.mean().reset_index()
        g = sp.groupby("config").a_kill.agg(["min", "max"])
        g["极差"] = (g["max"] - g["min"]).round(1)
        g["相对极差%"] = ((g["max"] - g["min"]) / g["max"] * 100).round(1)
        print(f"\n--- {tier} ---")
        print(g.round(1).to_string())

    if args.kind == "grouping":
        print("\n" + "=" * 90)
        print("[4] C3 诊断：投影前跨群指派数 / C2 撤单数（每场景均值）")
        print("=" * 90)
        print(summ[["cross", "dropped"]].loc[
            (slice(None), [n for n in names if n != "none"], slice(None))].to_string())

    print(f"\n产出 -> {OUT}")


if __name__ == "__main__":
    main()
