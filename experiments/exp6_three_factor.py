"""exp6 · 三因子主实验：算法 × 场景 × 编制档位（任务 P3.1–P3.4）。

把三份 exp1 结果合成一张长表，并产出可直接进技术报告的图表：

1. `metrics.csv`            —— 三因子长表（pandas 可直接分组）
2. `summary.csv`            —— (tier, algo) 的指标均值
3. `scores_by_tier.csv`     —— 方案 A 总分（含显著性）
4. `rank_matrix.csv`        —— 各指标在各档位下的平均秩
5. `cd_<tier>_<metric>.png` —— Nemenyi 临界差异图
6. `pareto_<tier>.png`      —— b–d 平面帕累托前沿
7. `rank_bump.png`          —— 名次随档位变化
8. `tier_metrics.png`       —— a/b/gap 随档位变化

用法：
    python experiments/exp6_three_factor.py
    python experiments/exp6_three_factor.py --tiers easy normal hard --metric a_gain
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm_alloc.config import ROOT  # noqa: E402
from swarm_alloc.evaluation.score import (DEFAULT_WEIGHTS, METRICS,  # noqa: E402
                                          nemenyi_cd, score_pareto,
                                          score_relative, significance_test)

# 档位 → 结果目录（easy 与历史 exp1 共用目录）
TIER_DIRS = {"easy": "exp1", "normal": "exp1_normal", "hard": "exp1_hard"}
OUT = ROOT / "results" / "three_factor"

# 出图统一用英文标签（AI-HANDOFF 约定：避免 CJK 字体缺方块）
SHORT = {"a_gain": "a", "b_threat": "b", "d_cost": "d", "time_s": "e"}


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------

def load_all(tiers) -> pd.DataFrame:
    frames = []
    for tier in tiers:
        d = ROOT / "results" / TIER_DIRS[tier]
        csv = d / "exp1_metrics.csv"
        if not csv.exists():
            raise FileNotFoundError(
                f"缺少 {csv}；请先跑：python experiments/exp1_main.py --tier {tier}")
        df = pd.read_csv(csv)
        df["tier"] = tier
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    order = {t: i for i, t in enumerate(tiers)}
    out["tier_order"] = out["tier"].map(order)
    return out.sort_values(["tier_order", "algo", "seed"]).drop(columns="tier_order")


def summarise(df: pd.DataFrame, tiers) -> pd.DataFrame:
    g = df.groupby(["tier", "algo"]).agg(
        a=("a_gain", "mean"), b=("b_threat", "mean"), d=("d_cost", "mean"),
        gap_a=("gap_a", "mean"), gap_b=("gap_b", "mean"),
        used=("used", "mean"), leaked=("leaked", "mean"),
        time=("time_s", "mean")).reset_index()
    g["tier"] = pd.Categorical(g["tier"], categories=tiers, ordered=True)
    return g.sort_values(["tier", "a"], ascending=[True, False])


# --------------------------------------------------------------------------
# 图
# --------------------------------------------------------------------------

def plot_cd(avg_rank: pd.Series, cd: float, title: str, out: Path):
    """Nemenyi 临界差异图（简化版 Demšar CD diagram）。"""
    ranks = avg_rank.sort_values()
    k = len(ranks)
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.42 * k))
    lo, hi = 0.5, k + 0.5
    y0 = 0.0
    ax.hlines(y0, lo, hi, color="k", lw=1.2)
    for r in range(1, k + 1):
        ax.vlines(r, y0 - 0.06, y0 + 0.06, color="k", lw=1.0)
        ax.text(r, y0 - 0.20, str(r), ha="center", va="top", fontsize=9)
    # CD 标尺
    ax.hlines(y0 + 0.35, k - cd, k, color="#C0504D", lw=3)
    ax.text(k - cd / 2, y0 + 0.44, f"CD = {cd:.2f}", ha="center", fontsize=9,
            color="#C0504D")
    # 算法
    for i, (name, r) in enumerate(ranks.items()):
        y = y0 + 0.9 + 0.42 * i
        ax.plot([r], [y], "o", color="#3C6E9B", ms=7)
        ax.text(r + 0.12, y, f"{name} ({r:.2f})", va="center", fontsize=9)
    # 把差异 < CD 的算法连成一组
    groups, cur = [], [list(ranks.index)[0]]
    for a, b in zip(ranks.index[:-1], ranks.index[1:]):
        if ranks[b] - ranks[a] < cd:
            cur.append(b)
        else:
            groups.append(cur)
            cur = [b]
    groups.append(cur)
    for gi, grp in enumerate(groups):
        y = y0 + 0.55 + 0.42 * (len(ranks) + gi)
        ax.hlines(y, ranks[grp[0]], ranks[grp[-1]], color="#4C9F70", lw=4,
                  alpha=0.75)
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(-0.5, y0 + 0.9 + 0.42 * (len(ranks) + len(groups)))
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=6)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_pareto(df: pd.DataFrame, tier: str, out: Path):
    """b–d 平面：每个点是一个 (算法, 场景)，粗点为算法均值。"""
    sub = df[df.tier == tier]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    algos = sorted(sub.algo.unique())
    cmap = plt.get_cmap("tab10")
    means = sub.groupby("algo")[["b_threat", "d_cost"]].mean()
    for i, a in enumerate(algos):
        g = sub[sub.algo == a]
        ax.scatter(g.b_threat, g.d_cost, s=8, alpha=0.25, color=cmap(i))
        ax.scatter([means.loc[a, "b_threat"]], [means.loc[a, "d_cost"]],
                   s=140, color=cmap(i), edgecolor="k", zorder=5,
                   label=f"{a}")
    # 均值点的帕累托前沿
    pts = means.sort_values("b_threat", ascending=False)
    front, best_d = [], -np.inf
    for a, r in pts.iterrows():
        if r.d_cost > best_d:
            front.append(a)
            best_d = r.d_cost
    fx = [means.loc[a, "b_threat"] for a in front]
    fy = [means.loc[a, "d_cost"] for a in front]
    ax.plot(fx, fy, "k--", lw=1.2, alpha=0.7,
            label="pareto front: " + ", ".join(front))
    ax.set_xlabel("b  threat elimination rate")
    ax.set_ylabel("d  value per sortie")
    ax.set_title(f"Pareto: threat elimination vs efficiency  (tier={tier})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_bump(rank_mat: pd.DataFrame, tiers, out: Path):
    """名次随档位变化（bump chart）。rank_mat: index=algo, columns=tier。"""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = np.arange(len(tiers))
    for algo in rank_mat.index:
        y = rank_mat.loc[algo, list(tiers)].to_numpy(dtype=float)
        ax.plot(x, y, "o-", lw=2, label=str(algo))
        for xi, yi in zip(x, y):
            ax.annotate(f"{int(yi)}", (xi, yi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(list(tiers))
    ax.invert_yaxis()
    ax.set_ylabel("rank (1 = best)")
    ax.set_title("Rank over fleet tiers (scheme A composite score)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_tier_metrics(summ: pd.DataFrame, tiers, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, col, title in [(axes[0], "a", "a: total expected value"),
                           (axes[1], "b", "b: threat elimination rate"),
                           (axes[2], "gap_a", "gap_a: optimality gap (lower better)")]:
        for algo, g in summ.groupby("algo"):
            ax.plot(range(len(tiers)), g[col].to_numpy(), "o-", label=algo)
        ax.set_xticks(range(len(tiers)))
        ax.set_xticklabels(tiers)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", default=["easy", "normal", "hard"])
    ap.add_argument("--metrics", nargs="+", default=list(METRICS))
    args = ap.parse_args()
    tiers = args.tiers
    OUT.mkdir(parents=True, exist_ok=True)

    df = load_all(tiers)
    df.to_csv(OUT / "metrics.csv", index=False)
    summ = summarise(df, tiers)
    summ.to_csv(OUT / "summary.csv", index=False)

    print("=" * 80)
    print("三因子主实验：算法 × 场景 × 档位")
    print("=" * 80)
    print(f"数据：{df.algo.nunique()} 算法 × {df.seed.nunique()} 场景 × {len(tiers)} 档位"
          f" = {len(df)} 行")
    print("\n[1] 指标均值")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ---- 方案 A 总分 ----
    print("\n" + "=" * 80)
    print("[2] 方案 A 总分（基线 greedy，threshold 6s）")
    print("=" * 80)
    score_cols, rank_cols = {}, {}
    for tier in tiers:
        sub = df[df.tier == tier][["algo", "seed", "a_gain", "b_threat",
                                   "d_cost", "time_s"]]
        a = score_relative(sub, DEFAULT_WEIGHTS, "greedy", "threshold", 6.0)
        score_cols[tier] = a["score"]
        rank_cols[tier] = a["rank"]
        print(f"\n--- {tier} ---")
        print(a[["r_a", "r_b", "r_d", "r_e", "score", "rank",
                 "sig_a", "sig_b", "sig_d"]].to_string(
            float_format=lambda x: f"{x:.4f}"))
    scores = pd.DataFrame(score_cols)
    ranks = pd.DataFrame(rank_cols)
    scores.to_csv(OUT / "scores_by_tier.csv")
    ranks.to_csv(OUT / "ranks_by_tier.csv")

    # ---- 排名稳定性（P3.4）----
    print("\n" + "=" * 80)
    print("[3] 排名稳定性（P3.4）")
    print("=" * 80)
    print(ranks.to_string())
    changed = (ranks.nunique(axis=1) > 1).sum()
    print(f"\n  名次发生过变化的算法：{changed}/{len(ranks)}")
    for algo in ranks.index:
        seq = " → ".join(str(int(ranks.loc[algo, t])) for t in tiers)
        print(f"    {algo:<18} {seq}")
    top1 = [scores[t].idxmax() for t in tiers]
    print(f"\n  冠军序列：{' → '.join(top1)}"
          f"  （{'发生翻转' if len(set(top1)) > 1 else '保持稳定'}）")
    plot_bump(ranks, tiers, OUT / "rank_bump.png")
    plot_tier_metrics(summ, tiers, OUT / "tier_metrics.png")

    # ---- 显著性 + CD 图（P3.2）----
    print("\n" + "=" * 80)
    print("[4] Friedman / Nemenyi（P3.2）")
    print("=" * 80)
    rank_rows = []
    for tier in tiers:
        sub = df[df.tier == tier]
        for m in args.metrics:
            hb = METRICS[m][1]
            st = significance_test(sub, m, hb)
            cd_auto = nemenyi_cd(st["k"], st["N"])
            print(f"\n  [{tier}] {SHORT[m]}  {st['test']}  p={st['p_value']:.4g}  "
                  f"CD={st['cd']:.3f}")
            for algo, r in st["avg_rank"].items():
                print(f"      {algo:<18} 平均秩 {r:.2f}")
                rank_rows.append(dict(tier=tier, metric=m, algo=algo, avg_rank=r))
            plot_cd(st["avg_rank"], st["cd"],
                    f"Nemenyi CD  |  tier={tier}  metric={SHORT[m]}  "
                    f"p={st['p_value']:.2g}  CD={st['cd']:.2f}",
                    OUT / f"cd_{tier}_{SHORT[m]}.png")
            _ = cd_auto
    pd.DataFrame(rank_rows).to_csv(OUT / "rank_matrix.csv", index=False)

    # ---- 帕累托（P3.3）----
    print("\n" + "=" * 80)
    print("[5] 帕累托前沿（P3.3）")
    print("=" * 80)
    for tier in tiers:
        b = score_pareto(df[df.tier == tier])
        print(f"  {tier:<8} b×d → {b['pareto']}")
        for plane, front in b["pareto_by_plane"].items():
            print(f"           {plane:<22} {front}")
        plot_pareto(df, tier, OUT / f"pareto_{tier}.png")

    print(f"\n产出 -> {OUT}")


if __name__ == "__main__":
    main()
