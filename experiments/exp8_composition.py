"""exp8 · 敌机配比灵敏度（任务 P3.5 第三组）。

思路
----
侦察型占比 30%、配额 5 架、单架次期望收益仅 0.40，是"歼灭价值最大化"与
"编队清空"两种诉求冲突的唯一来源。本实验把侦察型占比按倍数 f 缩放
（其余三类按比例重分配，总和保持 1），再**重新生成场景**并跑全部算法：

- f < 1 → 侦察机变少 → 每目标平均配额下降 → 兵力相对宽松
- f > 1 → 侦察机变多 → 每目标平均配额上升 → 兵力相对紧张

关键性质：`gen_wave` 里位置先生成、类型后抽样，所以**改配比不会改变敌机与
我方的位置**，只改变类型（价值、p_kill、配额）。因此不同 f 之间是配对可比的。

用法：
    python experiments/exp8_composition.py --scenes 30 --factors 0.5 1.0 1.5
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import swarm_alloc.algorithms  # noqa: F401,E402
from swarm_alloc.algorithms.base import available, get  # noqa: E402
from swarm_alloc.config import ROOT, load  # noqa: E402
from swarm_alloc.evaluation.oracle import scene_upper_bound  # noqa: E402
from swarm_alloc.evaluation.score import score_relative  # noqa: E402
from swarm_alloc.model import Model  # noqa: E402
from swarm_alloc.scenario.generator import gen_scene, tier_ratios  # noqa: E402
from swarm_alloc.simulation.simulate import simulate  # noqa: E402

OUT = ROOT / "results" / "exp8"


def scale_recon(etcfg: dict, factor: float) -> dict:
    """按倍数缩放侦察型占比，其余三类按比例重分配，总和保持 1。"""
    cfg = copy.deepcopy(etcfg)
    base = {t["name"]: float(t["ratio"]) for t in cfg["types"]}
    new = dict(base)
    new["recon"] = base["recon"] * factor
    others = [n for n in base if n != "recon"]
    rest_target = 1.0 - new["recon"]
    rest_sum = sum(base[n] for n in others)
    for n in others:
        new[n] = base[n] / rest_sum * rest_target
    for t in cfg["types"]:
        t["ratio"] = new[t["name"]]
    return cfg


def expected_quota(total_value: float, n: int, cfg: dict, model) -> float:
    """每目标平均配额需求（按配比加权）。"""
    acc = 0.0
    for t in cfg["types"]:
        m = model.quota(float(t["p_kill"]), model.kill_threshold)
        acc += float(t["ratio"]) * m
    return acc


def build_scenes(scfg, cfg_types, fcfg, ratio, n_scenes):
    return [gen_scene(seed, i, scfg, cfg_types, fcfg, ratio=ratio)
            for i, seed in enumerate(scfg["seeds"])][:n_scenes]


def run_variant(scenes, algo_names, model, scfg, cfg):
    rows = []
    for s in scenes:
        ob = scene_upper_bound(s, model)
        for name in algo_names:
            r = simulate(s, get(name), model, scfg, cfg)
            rows.append(dict(algo=name, seed=s.seed, n_pool0=len(s.friendly),
                             a_kill=r.a_kill, a_gain=r.a_gain,
                             b_threat=r.b_threat, d_cost=r.d_cost,
                             time_s=r.time_s, used=r.n_used,
                             destroyed=r.n_destroyed, leaked=r.n_leaked,
                             lost=r.n_lost, a_kill_opt=ob.a_kill_opt,
                             gap_a_kill=(ob.a_kill_opt - r.a_kill) / ob.a_kill_opt))
    return pd.DataFrame(rows)


def plot_scores(sc: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for algo, g in sc.groupby("algo"):
        g = g.sort_values("factor")
        ax.plot(g.factor, g.score, "o-", label=str(algo))
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("recon share multiplier (1.0 = 30%)")
    ax.set_ylabel("scheme-A score (baseline=1.0)")
    ax.set_title("Sensitivity to enemy composition")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=30)
    ap.add_argument("--factors", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    ap.add_argument("--tier", default="normal")
    ap.add_argument("--algos", nargs="+", default=None)
    args = ap.parse_args()

    scfg, etcfg, fcfg, ecfg = (load("scenario"), load("enemy_types"),
                               load("friendly"), load("eval"))
    model = Model(ecfg)
    ratio = tier_ratios(scfg)[args.tier]
    algo_names = args.algos or available()
    OUT.mkdir(parents=True, exist_ok=True)

    frames = []
    meta = []
    for f in args.factors:
        cfg = scale_recon(etcfg, f)
        scenes = build_scenes(scfg, cfg["types"], fcfg, ratio, args.scenes)
        shares = {t["name"]: round(float(t["ratio"]), 4) for t in cfg["types"]}
        eq = expected_quota(0.0, 0, cfg, model)
        vals = {t["name"]: float(t["value"]) for t in cfg["types"]}
        ev = sum(float(t["ratio"]) * vals[t["name"]] for t in cfg["types"])
        print(f"[run] recon×{f}: 占比={shares} 平均配额/目标={eq:.3f} "
              f"平均价值/目标={ev:.3f} 场景={len(scenes)}", flush=True)

        df = run_variant(scenes, algo_names, model, scfg, cfg)
        df["factor"] = f
        df["recon_share"] = float(cfg["types"][0]["ratio"])
        df["quota_per_target"] = eq
        frames.append(df)
        meta.append(dict(factor=f, recon_share=float(cfg["types"][0]["ratio"]),
                         quota_per_target=eq, value_per_target=ev, **shares))

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / "composition.csv", index=False)
    pd.DataFrame(meta).to_csv(OUT / "composition_meta.csv", index=False)

    summ = df.groupby(["factor", "algo"]).agg(
        a_kill=("a_kill", "mean"), b=("b_threat", "mean"),
        d=("d_cost", "mean"), used=("used", "mean"),
        leaked=("leaked", "mean"), gap=("gap_a_kill", "mean")).reset_index()
    summ.to_csv(OUT / "composition_summary.csv", index=False)

    sc = []
    for f, sub in df.groupby("factor"):
        a = score_relative(sub[["algo", "seed", "a_kill", "b_threat",
                                "d_cost", "time_s"]])
        for algo, row in a.iterrows():
            sc.append(dict(factor=f, algo=algo, score=row["score"],
                           rank=int(row["rank"])))
    sc = pd.DataFrame(sc)
    sc.to_csv(OUT / "composition_scores.csv", index=False)

    print("\n" + "=" * 78)
    print("[1] 指标均值")
    print("=" * 78)
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 78)
    print("[2] 方案 A 总分")
    print("=" * 78)
    print(sc.pivot_table(index="factor", columns="algo", values="score")
          .to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 78)
    print("[3] 冠军稳定性")
    print("=" * 78)
    for f, g in sc.groupby("factor"):
        print(f"  recon×{f}: 冠军 = {g.loc[g.score.idxmax(), 'algo']}")

    print("\n" + "=" * 78)
    print("[4] greedy 与 ga_iap 的主得分项对比（配对）")
    print("=" * 78)
    for f, sub in df.groupby("factor"):
        p = sub.pivot_table(index="seed", columns="algo", values="a_kill")
        if not {"greedy", "ga_iap"} <= set(p.columns):
            continue
        d = p["ga_iap"] - p["greedy"]
        sd = d.std(ddof=1)
        print(f"  recon×{f}: greedy={p['greedy'].mean():.2f} "
              f"ga_iap={p['ga_iap'].mean():.2f} 差={d.mean():+.2f} "
              f"d={(d.mean()/sd if sd>0 else 0):+.2f} 胜={int((d>0).sum())}/{len(d)}")

    plot_scores(sc, OUT / "composition_scores.png")
    print(f"\n产出 -> {OUT}")


if __name__ == "__main__":
    main()
