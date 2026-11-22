"""时间可行性反算：给定"距离 / 速度 / 升空高度 / 战局时长"，算"能拦住多少敌机"。

背景（老师建议 2026-11-17，V2-P0-4）
------------------------------------
新方案要求：我方从地面发射平台起飞，**先垂直升空 30 m**，再直线飞向目标。
于是"能否在敌机突防前到达"变成一个真实的物理问题：

    到达所需时间  T_fly   = climb_h / v_climb + d_min / v_cruise
    敌机突防剩余  T_until = (|r_j| - 防御半径) / v_j

    T_fly <= T_until  → 这一架敌机"理论上可拦截"

本脚本不依赖仿真，只做几何 + 运动学核算，用于**在改代码之前**判断参数组合是否自洽。

用法::

    python scripts/time_feasibility.py
    python scripts/time_feasibility.py --climb-h 30 --climb-speed 20
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm_alloc.config import ROOT, load            # noqa: E402
from swarm_alloc.scenario.generator import scene_file  # noqa: E402

OUT = ROOT / "results" / "feasibility"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", default=["normal"])
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--climb-h", type=float, default=30.0, help="垂直升空高度（米）")
    ap.add_argument("--climb-speed", type=float, default=20.0, help="垂直升空速度（m/s）")
    ap.add_argument("--cruise", nargs="+", type=float,
                    default=[30, 60, 100, 150, 200, 300, 500])
    args = ap.parse_args()

    scfg = load("scenario")
    defense_r = scfg["battlefield"]["defense_radius_m"]
    OUT.mkdir(parents=True, exist_ok=True)

    t_climb = args.climb_h / max(args.climb_speed, 1e-9)
    rows = []

    for tier in args.tiers:
        with open(ROOT / "data" / scene_file(tier, scfg), "rb") as f:
            scenes = pickle.load(f)[: args.scenes]

        for sc in scenes:
            # 我方：内球面上的一点（旧方案）；新方案里是平台上方 30 m，径向位置近似不变
            fpos = np.array([u.pos for u in sc.friendly])
            fpos = fpos + np.array([0.0, 0.0, args.climb_h])      # 升空后的起点

            for wi, wave in enumerate(sc.waves):
                epos = np.array([u.pos for u in wave.units])
                evel = np.array([u.vel for u in wave.units])
                r = np.linalg.norm(epos, axis=1)
                speed = np.linalg.norm(evel, axis=1)
                # 敌机突防剩余时间（假设它直线奔球心）
                t_until = (r - defense_r) / np.maximum(speed, 1e-9)

                # 每架敌机到"最近我机"的距离
                d = np.linalg.norm(epos[:, None, :] - fpos[None, :, :], axis=2)
                d_min = d.min(axis=1)

                for v in args.cruise:
                    t_fly = t_climb + d_min / v
                    ok = t_fly <= t_until
                    rows.append(dict(
                        tier=tier, seed=sc.seed, wave=wi, v_cruise=v,
                        n=len(epos), ok=int(ok.sum()),
                        r_med=float(np.median(r)), d_min_med=float(np.median(d_min)),
                        t_until_med=float(np.median(t_until)),
                        t_fly_med=float(np.median(t_fly)),
                    ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "time_feasibility_raw.csv", index=False)

    g = df.groupby(["v_cruise"]).agg(
        n=("n", "sum"), ok=("ok", "sum"),
        d_min_med=("d_min_med", "median"),
        t_until_med=("t_until_med", "median"),
        t_fly_med=("t_fly_med", "median"),
    )
    g["可拦截比例%"] = (g.ok / g.n * 100).round(1)
    g = g[["d_min_med", "t_until_med", "t_fly_med", "n", "ok", "可拦截比例%"]].round(1)

    print("=" * 88)
    print(f"升空高度 {args.climb_h:.0f} m ÷ 升空速度 {args.climb_speed:.0f} m/s = 升空耗时 {t_climb:.2f} s")
    print(f"战局时长 T = R/50；防御半径 {defense_r:.0f} m；场景 {args.scenes} 个 × {len(args.tiers)} 档")
    print("=" * 88)
    print(g.to_string())
    print()

    # 反推：要达到目标可拦截比例，需要多快
    print("=" * 88)
    print("【反推】达到给定可拦截比例所需的最小巡航速度")
    print("=" * 88)
    for target in (50, 80, 90, 95, 99):
        cand = g[g["可拦截比例%"] >= target]
        if len(cand):
            print(f"  ≥{target:2d}% → v_cruise ≥ {cand.index.min():.0f} m/s")
        else:
            print(f"  ≥{target:2d}% → 测试范围内（≤{max(args.cruise):.0f} m/s）达不到")
    print()
    print(f"产出 -> {OUT}")


if __name__ == "__main__":
    main()
