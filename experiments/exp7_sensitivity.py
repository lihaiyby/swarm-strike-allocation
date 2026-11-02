"""exp7 · 灵敏度分析（任务 P3.5）。

扫两个最可能动摇结论的参数：

1. **歼灭阈值 `kill_threshold`**（默认 0.9）
   它同时决定：配额 m_j（→ 总需求）、b 的歼灭判据、a_kill 的判据。
   阈值调高 → 配额变大 → 兵力相对变紧。预计会放大算法差异。

2. **敌方杀伤强度**（默认 q1=0.2 点杀伤、q2=0.5 面杀伤）
   它决定我方战损，进而影响资源池与重分配轮次。

每个配置都重算 oracle（上界随阈值变）与方案 A 总分，用于判断
「冠军是否稳定」「效应量是否稳定」。

用法：
    python experiments/exp7_sensitivity.py --scenes 30 --tier normal
"""
from __future__ import annotations

import argparse
import copy
import pickle
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
from swarm_alloc.scenario.generator import scene_file  # noqa: E402
from swarm_alloc.simulation.simulate import simulate  # noqa: E402

OUT = ROOT / "results" / "exp7"


def scale_lethality(etcfg: dict, scale: float) -> dict:
    """按倍数缩放所有杀伤参数 q（不改变半径与模式）。"""
    cfg = copy.deepcopy(etcfg)
    for t in cfg["types"]:
        leth = t.get("lethality")
        if leth:
            leth["q"] = float(np.clip(leth["q"] * scale, 0.0, 1.0))
    return cfg


def run_config(scenes, algo_names, model, scfg, etcfg, tag, param, value):
    rows = []
    for s in scenes:
        ob = scene_upper_bound(s, model)
        for name in algo_names:
            r = simulate(s, get(name), model, scfg, etcfg)
            rows.append(dict(
                sweep=tag, param=param, value=value, algo=name, seed=s.seed,
                n_pool0=len(s.friendly), a_kill=r.a_kill, a_gain=r.a_gain,
                b_threat=r.b_threat, d_cost=r.d_cost, time_s=r.time_s,
                used=r.n_used, destroyed=r.n_destroyed, leaked=r.n_leaked,
                lost=r.n_lost, a_kill_opt=ob.a_kill_opt,
                gap_a_kill=(ob.a_kill_opt - r.a_kill) / ob.a_kill_opt))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["sweep", "value", "algo"]).agg(
        a_kill=("a_kill", "mean"), b=("b_threat", "mean"),
        d=("d_cost", "mean"), lost=("lost", "mean"),
        used=("used", "mean"), gap=("gap_a_kill", "mean")).reset_index()


def scores_by_value(df: pd.DataFrame) -> pd.DataFrame:
    """每个取值下按方案 A 算总分（基线 greedy）。"""
    out = []
    for (sweep, value), sub in df.groupby(["sweep", "value"]):
        try:
            a = score_relative(sub[["algo", "seed", "a_kill", "b_threat",
                                    "d_cost", "time_s"]])
        except ValueError:
            continue
        for algo, row in a.iterrows():
            out.append(dict(sweep=sweep, value=value, algo=algo,
                            score=row["score"], rank=int(row["rank"])))
    return pd.DataFrame(out)


def plot_sweep(sc: pd.DataFrame, out: Path):
    sweeps = sc.sweep.unique()
    fig, axes = plt.subplots(1, len(sweeps), figsize=(6.4 * len(sweeps), 4.2),
                             squeeze=False)
    for ax, sw in zip(axes[0], sweeps):
        sub = sc[sc.sweep == sw]
        for algo, g in sub.groupby("algo"):
            g = g.sort_values("value")
            ax.plot(g.value, g.score, "o-", label=str(algo))
        ax.set_title(sw, fontsize=10)
        ax.set_xlabel("parameter value")
        ax.set_ylabel("scheme-A score (baseline=1.0)")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("Sensitivity of composite score to model parameters")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=30)
    ap.add_argument("--tier", default="normal")
    ap.add_argument("--thresholds", nargs="+", type=float,
                    default=[0.80, 0.85, 0.90, 0.95])
    ap.add_argument("--qscales", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--algos", nargs="+", default=None)
    args = ap.parse_args()

    scfg, etcfg, ecfg = load("scenario"), load("enemy_types"), load("eval")
    with open(ROOT / "data" / scene_file(args.tier, scfg), "rb") as f:
        scenes = pickle.load(f)[: args.scenes]
    algo_names = args.algos or available()
    OUT.mkdir(parents=True, exist_ok=True)

    frames = []

    # ---- 扫 1：歼灭阈值 ----
    for thr in args.thresholds:
        model = Model({**ecfg, "kill_threshold": thr})
        print(f"[run] kill_threshold={thr} × {len(scenes)} 场景 × {len(algo_names)} 算法",
              flush=True)
        frames.append(run_config(scenes, algo_names, model, scfg, etcfg,
                                 "kill_threshold", "kill_threshold", thr))

    # ---- 扫 2：杀伤强度 ----
    model = Model(ecfg)
    for qs in args.qscales:
        cfg2 = scale_lethality(etcfg, qs)
        print(f"[run] lethality_scale={qs} × {len(scenes)} 场景", flush=True)
        frames.append(run_config(scenes, algo_names, model, scfg, cfg2,
                                 "lethality_scale", "q_scale", qs))

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / "sensitivity.csv", index=False)
    summ = summarise(df)
    summ.to_csv(OUT / "sensitivity_summary.csv", index=False)
    sc = scores_by_value(df)
    sc.to_csv(OUT / "sensitivity_scores.csv", index=False)

    print("\n" + "=" * 78)
    print("[1] 指标均值")
    print("=" * 78)
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 78)
    print("[2] 方案 A 总分（基线 greedy）")
    print("=" * 78)
    print(sc.pivot_table(index=["sweep", "value"], columns="algo",
                         values="score").to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 78)
    print("[3] 冠军稳定性")
    print("=" * 78)
    for sw, g in sc.groupby("sweep"):
        champs = []
        for value, gg in g.groupby("value"):
            champs.append(f"{value}→{gg.loc[gg.score.idxmax(), 'algo']}")
        print(f"  {sw:<18} " + "  ".join(champs))

    plot_sweep(sc, OUT / "sensitivity_scores.png")
    print(f"\n产出 -> {OUT}")


if __name__ == "__main__":
    main()
