"""exp9 · 威胁度定义灵敏度（任务 P3.5 第四组）。

威胁度 `w_k` 在仿真里有两个作用：
1. `Model.select_groups` 按威胁度从高到低纳入群（资源不足时决定放弃谁）；
2. `greedy` 按威胁度顺序处理群并取机。

本实验替换感知层的威胁度算法，比较四种定义：

| 变体 | 定义 |
|---|---|
| `entropy`（现状） | 熵权 TOPSIS，三特征：群总价值、平均速度、到基地最近距离 |
| `equal` | 等权 TOPSIS（不用熵权） |
| `value` | 直接用群总价值归一化（单一特征） |
| `random` | 每群一个确定性伪随机数（阈值下界：威胁排序失效） |

实现方式：**运行时替换 `swarm_alloc.simulation.simulate.topsis`**，
其余仿真逻辑完全不变（`simulate` 在调用时查全局名）。

用法：
    python experiments/exp9_threat.py --scenes 30 --tier normal
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

import swarm_alloc.algorithms  # noqa: F401,E402
import swarm_alloc.simulation.simulate as simmod  # noqa: E402
from swarm_alloc.algorithms.base import available, get  # noqa: E402
from swarm_alloc.config import ROOT, load  # noqa: E402
from swarm_alloc.evaluation.oracle import scene_upper_bound  # noqa: E402
from swarm_alloc.evaluation.score import score_relative  # noqa: E402
from swarm_alloc.model import Model  # noqa: E402
from swarm_alloc.perception.topsis import entropy_weights  # noqa: E402
from swarm_alloc.scenario.generator import scene_file  # noqa: E402
from swarm_alloc.simulation.simulate import simulate  # noqa: E402

OUT = ROOT / "results" / "exp9"
ORIG_TOPSIS = simmod.topsis


# --------------------------------------------------------------------------
# 四种威胁度定义
# --------------------------------------------------------------------------

def _features(groups, units):
    feats = []
    for g in groups:
        us = [units[i] for i in g.member_ids]
        total_value = sum(u.value for u in us)
        mean_speed = float(np.mean([u.speed for u in us]))
        min_dist = min(float(np.linalg.norm(u.pos)) for u in us)
        feats.append([total_value, mean_speed, min_dist])
    return np.array(feats, dtype=float)


def topsis_entropy(groups, units):
    """现状：熵权 + TOPSIS（三特征）。"""
    return ORIG_TOPSIS(groups, units)


def topsis_equal(groups, units):
    """等权 TOPSIS：跳过熵权法，三个特征各占 1/3。"""
    if not groups:
        return groups
    M = _features(groups, units)
    M_pos = np.column_stack([M[:, 0], M[:, 1], -M[:, 2]])   # 前两个正向，距离取负
    w = np.full(M_pos.shape[1], 1.0 / M_pos.shape[1])
    V = M_pos / (np.linalg.norm(M_pos, axis=0, keepdims=True) + 1e-12) * w
    d_best = np.linalg.norm(V - V.max(axis=0), axis=1)
    d_worst = np.linalg.norm(V - V.min(axis=0), axis=1)
    score = d_worst / (d_best + d_worst + 1e-12)
    for g, s in zip(groups, score):
        g.threat = float(s)
    return groups


def topsis_value(groups, units):
    """只用群总价值：threat = 该群价值 / 全场群价值之和。"""
    if not groups:
        return groups
    vals = np.array([sum(units[i].value for i in g.member_ids) for g in groups],
                    dtype=float)
    tot = vals.sum()
    for g, v in zip(groups, vals / (tot + 1e-12)):
        g.threat = float(v)
    return groups


def topsis_random(groups, units):
    """确定性伪随机威胁度：用于检验"威胁排序完全失效"时的下界。"""
    if not groups:
        return groups
    for g in groups:
        seed = (int(g.id) * 1000003 + len(g.member_ids) * 7919) % (2 ** 32)
        g.threat = float(np.random.default_rng(seed).random())
    return groups


VARIANTS = {
    "entropy": (topsis_entropy, "熵权 TOPSIS（现状）"),
    "equal": (topsis_equal, "等权 TOPSIS"),
    "value": (topsis_value, "仅群总价值"),
    "random": (topsis_random, "确定性随机（排序失效）"),
}


# --------------------------------------------------------------------------

def run_variant(scenes, algo_names, model, scfg, etcfg, tag, label):
    rows = []
    for s in scenes:
        ob = scene_upper_bound(s, model)
        for name in algo_names:
            r = simulate(s, get(name), model, scfg, etcfg)
            rows.append(dict(variant=tag, label=label, algo=name, seed=s.seed,
                             n_pool0=len(s.friendly), a_kill=r.a_kill,
                             a_gain=r.a_gain, b_threat=r.b_threat,
                             d_cost=r.d_cost, time_s=r.time_s, used=r.n_used,
                             destroyed=r.n_destroyed, leaked=r.n_leaked,
                             lost=r.n_lost, a_kill_opt=ob.a_kill_opt,
                             gap_a_kill=(ob.a_kill_opt - r.a_kill) / ob.a_kill_opt))
    return pd.DataFrame(rows)


def plot_scores(sc: pd.DataFrame, out: Path):
    order = list(VARIANTS)
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    x = np.arange(len(order))
    for algo, g in sc.groupby("algo"):
        y = [float(g[g.variant == v]["score"].iloc[0]) for v in order]
        ax.plot(x, y, "o-", label=str(algo))
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("scheme-A score (baseline=1.0)")
    ax.set_title("Sensitivity to threat-evaluation definition")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=30)
    ap.add_argument("--tier", default="normal")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--algos", nargs="+", default=None)
    args = ap.parse_args()

    scfg, etcfg, ecfg = load("scenario"), load("enemy_types"), load("eval")
    model = Model(ecfg)
    with open(ROOT / "data" / scene_file(args.tier, scfg), "rb") as f:
        scenes = pickle.load(f)[: args.scenes]
    algo_names = args.algos or available()
    OUT.mkdir(parents=True, exist_ok=True)

    frames = []
    for tag in args.variants:
        fn, label = VARIANTS[tag]
        simmod.topsis = fn                      # ← 运行时替换
        print(f"[run] 威胁度={tag}（{label}）× {len(scenes)} 场景", flush=True)
        frames.append(run_variant(scenes, algo_names, model, scfg, etcfg, tag, label))
    simmod.topsis = ORIG_TOPSIS                # 还原，避免影响后续

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / "threat.csv", index=False)

    summ = df.groupby(["variant", "algo"]).agg(
        a_kill=("a_kill", "mean"), b=("b_threat", "mean"),
        d=("d_cost", "mean"), used=("used", "mean"),
        leaked=("leaked", "mean"), gap=("gap_a_kill", "mean")).reset_index()
    summ.to_csv(OUT / "threat_summary.csv", index=False)

    sc = []
    for tag, sub in df.groupby("variant"):
        a = score_relative(sub[["algo", "seed", "a_kill", "b_threat",
                                "d_cost", "time_s"]])
        for algo, row in a.iterrows():
            sc.append(dict(variant=tag, algo=algo, score=row["score"],
                           rank=int(row["rank"])))
    sc = pd.DataFrame(sc)
    sc.to_csv(OUT / "threat_scores.csv", index=False)

    print("\n" + "=" * 78)
    print("[1] 指标均值")
    print("=" * 78)
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 78)
    print("[2] 方案 A 总分")
    print("=" * 78)
    print(sc.pivot_table(index="variant", columns="algo", values="score")
          .to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 78)
    print("[3] 冠军与 greedy−ga_iap 主项差")
    print("=" * 78)
    for tag, g in sc.groupby("variant"):
        sub = df[df.variant == tag].pivot_table(index="seed", columns="algo",
                                                values="a_kill")
        d = sub["ga_iap"] - sub["greedy"]
        sd = d.std(ddof=1)
        print(f"  {tag:<8} 冠军={g.loc[g.score.idxmax(), 'algo']:<10} "
              f"ga_iap−greedy={d.mean():+7.2f}  d={(d.mean()/sd if sd>0 else 0):+5.2f}  "
              f"胜={int((d>0).sum())}/{len(d)}")

    plot_scores(sc, OUT / "threat_scores.png")
    print(f"\n产出 -> {OUT}")


if __name__ == "__main__":
    main()
