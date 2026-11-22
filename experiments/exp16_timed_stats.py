"""exp16 · 时序口径结果的显著性分析（V2-P2 配套）。

对 `exp15_timed_main.py` 产出的三档 raw 表做：
1. **Friedman 整体检验**：12 个算法在关键指标上是否整体有差异；
2. **Nemenyi 临界差异 CD**：名次差多少才算真的差；
3. **两两 Wilcoxon + Holm 校正**：给出"冠军 vs 亚军"这类对比的 p 值。

用法::

    python experiments/exp16_timed_stats.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm_alloc.evaluation.score import nemenyi_cd  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "timed"
TIERS = ["easy", "normal", "hard"]
METRICS = {"a_kill": True, "b_threat": True, "d_cost": True}


def _holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for r, idx in enumerate(order):
        val = min(1.0, (m - r) * float(pvals[idx]))
        run = max(run, val)
        adj[idx] = run
    return adj


def main():
    from scipy.stats import friedmanchisquare, wilcoxon

    for tier in TIERS:
        f = OUT / f"timed_{tier}_raw.csv"
        if not f.exists():
            print(f"[跳过] {tier}：缺 {f.name}")
            continue
        df = pd.read_csv(f)
        piv = {a: g.set_index("seed").sort_index() for a, g in df.groupby("algo")}
        algos = sorted(piv)
        n = len(next(iter(piv.values())))
        k = len(algos)

        print("=" * 100)
        print(f"【{tier}】{k} 个算法 × {n} 个场景")
        print("=" * 100)

        for metric, hb in METRICS.items():
            mat = np.vstack([piv[a][metric].to_numpy() for a in algos])
            stat, p = friedmanchisquare(*mat)
            if hb:                      # 越大越好 → 名次 1 最好
                ranks = (-mat).argsort(axis=0).argsort(axis=0) + 1
            else:
                ranks = mat.argsort(axis=0).argsort(axis=0) + 1
            avg = pd.Series(ranks.mean(axis=1), index=algos).sort_values()
            cd = nemenyi_cd(k, n)
            print(f"\n-- {metric} --  Friedman χ²={stat:.1f}  p={p:.3g}   CD={cd:.3f}")
            print("   平均名次: " + ", ".join(f"{a}={v:.2f}" for a, v in avg.items()))

            top = list(avg.index[:6])
            pairs, ps = [], []
            for a, b in itertools.combinations(top, 2):
                x = piv[a][metric].to_numpy()
                y = piv[b][metric].to_numpy()
                try:
                    ps.append(wilcoxon(x, y).pvalue)
                except ValueError:
                    ps.append(1.0)
                pairs.append((a, b))
            adj = _holm(ps)
            print("   前 6 名两两 Wilcoxon（Holm 校正）:")
            for (a, b), p0, p1 in zip(pairs, ps, adj):
                star = "***" if p1 < 1e-3 else "**" if p1 < 1e-2 else "*" if p1 < 0.05 else "n.s."
                print(f"     {a:20s} vs {b:20s} p={p0:.3g} → Holm {p1:.3g} {star}")

        # 综合分的稳定性：均值 ± 标准差
        if (OUT / f"timed_{tier}_score.csv").exists():
            sc = pd.read_csv(OUT / f"timed_{tier}_score.csv", index_col=0)
            print("\n   综合分（方案 A）前 6:")
            print(sc.sort_values("score", ascending=False).head(6)[["score", "rank"]].round(4).to_string())
        print()


if __name__ == "__main__":
    main()
