"""总分接口层：三套并行的评估口径（任务 P3.6）。

为什么要有三套？
----------------
`config/eval.yaml` 里的权重 A~E 一直是 `null`，所以"总分"从未被真正计算过。
但直接加权求和有三个硬坑：

1. **量纲差 652 倍**：a≈590、b≈1、d≈1.7、e 从 0.006 s 到 3.9 s；
2. **方向不一致**：e 越小越好；
3. **c 不是逐场景指标**：它是跨 50 场景的标准差，无法与 a/b/d/e 并列相加。

因此本模块提供三种口径，各有各的用途：

| 方案 | 函数 | 产出 | 适用 |
|---|---|---|---|
| **A 相对基准分** | `score_relative()` | 每个算法一个标量（基准 = 1.000） | 主表，一眼看排名 |
| **B 不合成** | `score_pareto()` | 帕累托前沿 + 逐场景胜负 + 显著性检验 | 附录，诚实呈现取舍 |
| **C 风险调整分** | `score_risk_adjusted()` | 标量（A 加不稳定惩罚） | 强调稳定性的场景 |

设计约定
--------
- A 与 C 都以**基线算法（默认 greedy）逐场景的比值**为单位：天然无量纲、自动处理
  "越小越好"，且不依赖参赛算法数量（min-max 归一在只有 2 个算法时会退化成 0/1）。
- C = A 的逐场景比值再做 `mean − λ·std`；λ=0 时与 A 完全相同。
- 时间 e 的计分语义可切换，默认 `"threshold"`（对应 `方案04-v3 §7`「单次分配耗时 ≤1 s 不罚」）。

用法
----
    python -m swarm_alloc.evaluation.score                    # 用 results/exp1/exp1_metrics.csv
    python -m swarm_alloc.evaluation.score --time-mode ratio  # 换成线性折算对比
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# 指标列 → (中文名, 是否越大越好)
METRICS = {
    "a_gain": ("a 打击总收益", True),
    "b_threat": ("b 威胁消除率", True),
    "d_cost": ("d 效费比", True),
    "time_s": ("e 求解耗时", False),
}

# 默认权重：收益/教义侧 70%、效率侧 30%（D3 待拍板，此处仅为可运行默认值）
DEFAULT_WEIGHTS = {"a_gain": 0.35, "b_threat": 0.35, "d_cost": 0.20, "time_s": 0.10}
EQUAL_WEIGHTS = {k: 0.25 for k in METRICS}

# 时间计量模式
TIME_MODES = {
    "threshold": "耗时 ≤ limit 记 1.0，超出按 limit/耗时（对应方案04-v3 §7，默认）",
    "ratio": "r = 基线耗时 / 本算法耗时（线性，对数量级差极敏感）",
    "rank": "逐场景跨算法排名（最快 1.0，最慢 1/k）",
}

# Nemenyi 检验的 q_α（α=0.05，双侧；Demšar 2006）
_Q_ALPHA_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
               6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

# 逐场景比值如何汇聚成一个数（P3.7）
AGG_MODES = {
    "mean": "算术平均（默认）——对极端场景敏感",
    "median": "中位数——抗离群",
    "geomean": "几何平均——比值的标准做法，不受 Jensen 偏置",
    "ratio_of_means": "先求均值再相除（= Σa/Σa_base）",
}

_EPS = 1e-12


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------

def _pivot(df: pd.DataFrame) -> dict:
    """长表 → {algorithm: DataFrame(index=seed)}。"""
    return {a: g.set_index("seed").sort_index() for a, g in df.groupby("algo")}


def _normalize_weights(weights: dict | None) -> dict:
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    unknown = set(w) - set(METRICS)
    if unknown:
        raise ValueError(f"未知指标: {sorted(unknown)}；可选 {sorted(METRICS)}")
    total = sum(w.values())
    if total <= 0:
        raise ValueError("权重之和必须为正")
    return {k: v / total for k, v in w.items() if v > 0}


def _time_scores(piv: dict, baseline: str, mode: str, limit_s: float) -> pd.DataFrame:
    """e 的逐场景得分矩阵（index=seed, columns=algo），越大越好。"""
    times = pd.DataFrame({a: d["time_s"] for a, d in piv.items()})
    if mode == "threshold":
        return pd.DataFrame(
            np.where(times.to_numpy() <= limit_s, 1.0, limit_s / times.clip(lower=_EPS)),
            index=times.index, columns=times.columns)
    if mode == "ratio":
        base = times[baseline]
        return pd.DataFrame({a: base / times[a].clip(lower=_EPS) for a in times.columns},
                            index=times.index)
    if mode == "rank":
        r = times.rank(axis=1, method="average")
        k = times.shape[1]
        return 1.0 - (r - 1.0) / max(k - 1, 1)
    raise ValueError(f"未知 time_mode: {mode!r}；可选 {sorted(TIME_MODES)}")


def _ratio_frames(piv: dict, baseline: str, time_mode: str,
                  limit_s: float) -> dict:
    """返回 {algo: dict(指标列 → 逐场景比值 Series)}。"""
    base = piv[baseline]
    escore = _time_scores(piv, baseline, time_mode, limit_s)
    out = {}
    for algo, cur in piv.items():
        common = base.index.intersection(cur.index)
        r = {c: cur.loc[common, c] / base.loc[common, c].clip(lower=_EPS)
             for c in ("a_gain", "b_threat", "d_cost")}
        r["time_s"] = escore.loc[common, algo]
        out[algo] = r
    return out


def _aggregate(r: pd.Series, mode: str = "mean",
               num: pd.Series | None = None, den: pd.Series | None = None) -> float:
    """把一个逐场景比值序列汇聚成一个标量（P3.7）。"""
    if mode == "mean":
        return float(r.mean())
    if mode == "median":
        return float(r.median())
    if mode == "geomean":
        return float(np.exp(np.log(r.clip(lower=_EPS)).mean()))
    if mode == "ratio_of_means":
        if num is None or den is None:
            raise ValueError("ratio_of_means 需要提供 num/den 原始序列")
        return float(num.mean() / max(den.mean(), _EPS))
    raise ValueError(f"未知 agg: {mode!r}；可选 {sorted(AGG_MODES)}")


def _holm(pvals: list) -> list:
    """Holm–Bonferroni 校正（族错误率控制）。"""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * float(pvals[idx]))
        running = max(running, val)
        adj[idx] = running
    return adj.tolist()


def _star(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _sig_vs_baseline(piv: dict, baseline: str, metric: str,
                     higher_better: bool) -> dict:
    """各算法相对基线的 Wilcoxon 符号秩检验 p 值（Holm 校正）。"""
    from scipy.stats import wilcoxon

    base = piv[baseline]
    algos = sorted(piv)
    raw = []
    for a in algos:
        if a == baseline:
            raw.append(1.0)
            continue
        common = base.index.intersection(piv[a].index)
        x, y = piv[a].loc[common, metric], base.loc[common, metric]
        try:
            _, p = wilcoxon(x.to_numpy(), y.to_numpy())
            raw.append(float(p))
        except ValueError:                      # 全部相等，无法检验
            raw.append(1.0)
    adj = _holm(raw)
    return dict(zip(algos, adj))


def nemenyi_cd(k: int, n_scenes: int) -> float:
    """Nemenyi 临界差异 CD = q_α·√(k(k+1)/6N)（α=0.05，双侧）。"""
    if k < 2 or n_scenes <= 0:
        return float("nan")
    return float(_Q_ALPHA_05.get(k, 3.164) * np.sqrt(k * (k + 1) / (6.0 * n_scenes)))


# --------------------------------------------------------------------------
# 方案 A：相对基准分
# --------------------------------------------------------------------------

def score_relative(df: pd.DataFrame, weights: dict | None = None,
                   baseline: str = "greedy", time_mode: str = "threshold",
                   time_limit_s: float = 6.0, agg: str = "mean",
                   significance: bool = True) -> pd.DataFrame:
    """方案 A —— 以基线算法为 1.000 的相对分。

    逐场景先算 `r_i(s) = 指标_i(算法) / 指标_i(基线)`（e 按 `time_mode` 处理），
    再按 `agg` 汇聚（默认算术平均），最后按权重求和。

    参数
    ----
    agg : `mean` / `median` / `geomean` / `ratio_of_means`，见 `AGG_MODES`。
          比值型数据推荐 `geomean`（不受 Jensen 偏置）。
    significance : 为 True 时额外给出各指标相对基线的 Wilcoxon p 值（Holm 校正）
          与显著性星号列 `sig_a/sig_b/sig_d/sig_e`——**比值本身不含统计信息**，
          0.3% 的差距可能只是噪声（见 P3.7）。

    返回 DataFrame（index=algo）：`r_a`~`r_e`、`score`、`rank`、`n_scene`
    （开启 `significance` 时另有 `p_*` 与 `sig_*`）。
    """
    if agg not in AGG_MODES:
        raise ValueError(f"未知 agg: {agg!r}；可选 {sorted(AGG_MODES)}")
    w = _normalize_weights(weights)
    piv = _pivot(df)
    if baseline not in piv:
        raise ValueError(f"基线算法 {baseline!r} 不在数据中；可选 {sorted(piv)}")
    ratios = _ratio_frames(piv, baseline, time_mode, time_limit_s)

    rows = []
    for algo, r in ratios.items():
        common = r["time_s"].index
        vals = {}
        for c in METRICS:
            if agg == "ratio_of_means":
                if c == "time_s":
                    # e 不是"指标比"，无法用均值之比；退回算术平均
                    vals[c] = _aggregate(r[c], "mean")
                else:
                    vals[c] = _aggregate(r[c], agg,
                                         num=piv[algo].loc[common, c],
                                         den=piv[baseline].loc[common, c])
            else:
                vals[c] = _aggregate(r[c], agg)
        rows.append(dict(algo=algo,
                         r_a=vals["a_gain"], r_b=vals["b_threat"],
                         r_d=vals["d_cost"], r_e=vals["time_s"],
                         score=sum(w[c] * vals[c] for c in w),
                         n_scene=len(common)))
    out = pd.DataFrame(rows).set_index("algo")

    if significance:
        alias = {"a_gain": "a", "b_threat": "b", "d_cost": "d", "time_s": "e"}
        for m, (_, hb) in METRICS.items():
            ps = _sig_vs_baseline(piv, baseline, m, hb)
            out[f"p_{alias[m]}"] = [ps[a] for a in out.index]
            out[f"sig_{alias[m]}"] = [_star(ps[a]) for a in out.index]

    out["rank"] = out["score"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("score", ascending=False)


# --------------------------------------------------------------------------
# 方案 C：风险调整分
# --------------------------------------------------------------------------

def score_risk_adjusted(df: pd.DataFrame, weights: dict | None = None,
                        baseline: str = "greedy", lam: float = 1.0,
                        time_mode: str = "threshold",
                        time_limit_s: float = 6.0, agg: str = "mean") -> pd.DataFrame:
    """方案 C —— 方案 A 的逐场景比值再做 `mean − λ·std`。

    指标 c（稳定性）由此自然进入总分：波动大的算法被扣分，λ 控制惩罚强度。
    λ=0 且 `agg="mean"` 时结果与方案 A 完全相同。
    """
    if agg not in AGG_MODES:
        raise ValueError(f"未知 agg: {agg!r}；可选 {sorted(AGG_MODES)}")
    w = _normalize_weights(weights)
    piv = _pivot(df)
    ratios = _ratio_frames(piv, baseline, time_mode, time_limit_s)

    rows = []
    for algo, r in ratios.items():
        adj = {c: _aggregate(r[c], agg) - lam * float(r[c].std(ddof=1)) for c in METRICS}
        rows.append(dict(algo=algo,
                         adj_a=adj["a_gain"], adj_b=adj["b_threat"],
                         adj_d=adj["d_cost"], adj_e=adj["time_s"],
                         score=sum(w[c] * adj[c] for c in w),
                         lam=lam, n_scene=len(r["time_s"])))
    out = pd.DataFrame(rows).set_index("algo")
    out["rank"] = out["score"].rank(ascending=False, method="min").astype(int)
    return out.sort_values("score", ascending=False)


# --------------------------------------------------------------------------
# 方案 B：不合成
# --------------------------------------------------------------------------

def _pareto_front(points: pd.DataFrame) -> list:
    """points: index=algo，各列均为越大越好。返回非支配集合。"""
    front = []
    for a in points.index:
        dominated = False
        for b in points.index:
            if a == b:
                continue
            if (points.loc[b] >= points.loc[a]).all() and (points.loc[b] > points.loc[a]).any():
                dominated = True
                break
        if not dominated:
            front.append(a)
    return sorted(front)


def significance_test(df: pd.DataFrame, metric: str, higher_better: bool = True) -> dict:
    """单指标的两两显著性检验 + 平均秩 + Nemenyi 临界差异。

    k ≥ 3 用 Friedman；k = 2 时 Friedman 退化，自动改用 Wilcoxon 符号秩检验。
    """
    from scipy.stats import friedmanchisquare, wilcoxon

    piv = _pivot(df)
    algos = sorted(piv)
    seeds = None
    for a in algos:
        seeds = piv[a].index if seeds is None else seeds.intersection(piv[a].index)
    seeds = seeds.sort_values()

    mat = np.column_stack([
        piv[a].loc[seeds, metric].to_numpy() * (1 if higher_better else -1)
        for a in algos])
    k, N = len(algos), len(seeds)

    # 逐场景排名（1 = 最好），并列取平均
    rank_mat = np.zeros((N, k))
    for i in range(N):
        rank_mat[i] = pd.Series(mat[i]).rank(ascending=False, method="average").to_numpy()
    avg_rank = pd.Series(rank_mat.mean(axis=0), index=algos)

    test_name, stat, p = None, float("nan"), float("nan")
    try:
        if k >= 3:
            stat, p = friedmanchisquare(*[mat[:, j] for j in range(k)])
            test_name = "Friedman"
        elif k == 2:
            stat, p = wilcoxon(mat[:, 0], mat[:, 1])
            test_name = "Wilcoxon"
    except ValueError:
        test_name = "n/a"

    q = _Q_ALPHA_05.get(k, 3.164)
    cd = float(q * np.sqrt(k * (k + 1) / (6.0 * N))) if k >= 2 else float("nan")

    pairs = []
    for a, b in combinations(algos, 2):
        diff = abs(avg_rank[a] - avg_rank[b])
        pairs.append(dict(pair=f"{a} vs {b}", rank_diff=float(diff),
                          significant=bool(diff > cd)))

    return dict(metric=metric, test=test_name, avg_rank=avg_rank,
                statistic=float(stat), p_value=float(p), cd=cd,
                k=k, N=N, significant_pairs=pairs)


def score_pareto(df: pd.DataFrame, dims=("b_threat", "d_cost")) -> dict:
    """方案 B —— 不给单一总分，只给可比较的结构化证据。

    包含五件东西：
    1. `means`：各算法指标均值表；
    2. `pareto`：`dims` 指定平面上的非支配集合；
    3. `pareto_by_plane`：三组常用二维平面的前沿（a–d / b–d / b–e），
       用来暴露"**前沿依赖你选哪两个维度**"这件事；
    4. `win_matrix`：逐场景两两胜负次数（四指标合计）；
    5. `stats`：每个指标的显著性检验 + 平均秩 + Nemenyi CD。
    """
    piv = _pivot(df)
    means = pd.DataFrame({a: piv[a][list(METRICS)].mean() for a in piv}).T[list(METRICS)]

    def _front(axes):
        plane = pd.DataFrame(index=means.index)
        for d in axes:
            if d not in METRICS:
                raise ValueError(f"未知维度 {d!r}；可选 {sorted(METRICS)}")
            plane[d] = means[d] if METRICS[d][1] else -means[d]
        return _pareto_front(plane)

    planes = {("a_gain", "d_cost"): None, ("b_threat", "d_cost"): None,
              ("b_threat", "time_s"): None}
    pareto_by_plane = {f"{x} × {y}": _front((x, y)) for x, y in planes}

    algos = sorted(piv)
    win = pd.DataFrame(0, index=algos, columns=algos, dtype=int)
    for a in algos:
        for b in algos:
            if a == b:
                continue
            common = piv[a].index.intersection(piv[b].index)
            wins = 0
            for col, (_, hb) in METRICS.items():
                x, y = piv[a].loc[common, col], piv[b].loc[common, col]
                wins += int(((x > y) if hb else (x < y)).sum())
            win.loc[a, b] = wins

    stats = {m: significance_test(df, m, hb) for m, (_, hb) in METRICS.items()}
    return dict(means=means, pareto_dimensions=list(dims),
                pareto=pareto_by_plane[f"{dims[0]} × {dims[1]}"],
                pareto_by_plane=pareto_by_plane,
                win_matrix=win, stats=stats)


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------

def _fmt_ranks(s: pd.Series) -> str:
    return "{" + ", ".join(f"{k}: {v:.2f}" for k, v in s.items()) + "}"


def format_report(df: pd.DataFrame, baseline: str = "greedy",
                  weights: dict | None = None, time_mode: str = "threshold",
                  time_limit_s: float = 6.0, lam: float = 1.0,
                  agg: str = "mean") -> str:
    w = _normalize_weights(weights)
    L = []
    add = L.append
    add("=" * 78)
    add("总分接口 · 三套口径对比")
    add("=" * 78)
    alias = {"a_gain": "a", "b_threat": "b", "d_cost": "d", "time_s": "e"}
    wtxt = "  ".join(f"{alias[k]}={v:.2f}" for k, v in w.items())
    add(f"基线={baseline}   权重[{wtxt}]   时间模式={time_mode}(limit={time_limit_s}s)"
        f"   λ={lam}   agg={agg}")
    add(f"数据：{df.algo.nunique()} 个算法 × {df.seed.nunique()} 个场景")

    a = score_relative(df, weights, baseline, time_mode, time_limit_s, agg=agg)
    c = score_risk_adjusted(df, weights, baseline, lam, time_mode, time_limit_s, agg=agg)
    b = score_pareto(df)

    add("\n" + "-" * 78)
    add("方案 A · 相对基准分（基准 = 1.000，>1 优于基线）")
    add("   显著性： vs 基线的 Wilcoxon 符号秩检验，Holm 校正；***p<.001 **p<.01 *p<.05")
    add("-" * 78)
    base_cols = ["r_a", "r_b", "r_d", "r_e", "score", "rank"]
    view = a[base_cols].copy()
    for m, al in alias.items():
        view[f"sig_{al}"] = a[f"sig_{al}"]
    add(view.to_string(float_format=lambda x: f"{x:.4f}"))

    if time_mode == "threshold" and a["r_e"].nunique() == 1:
        add(f"\n  ⚠️ 所有算法均满足 {time_limit_s}s 门槛 → e 不再区分，但仍占 "
            f"{w.get('time_s', 0):.0%} 权重，会把所有分数向 1.0 压缩。"
            f"\n     若想让 e 参与区分，改用 --time-mode ratio 或调小 --time-limit。")
    if time_mode == "threshold":
        add(f"\n  ⚠️ 门槛 {time_limit_s}s 是**每场累计**耗时的人为取值；方案04-v3 的原文规则是"
            f"\"单轮 ≤1 s\"。\n     exp1 目前没记录逐轮耗时，故此处为近似——见任务 P3.6 备注。")

    add("\n" + "-" * 78)
    add("方案 C · 风险调整分（mean − λ·std；λ=0 且 agg=mean 时等于方案 A）")
    add("-" * 78)
    add(c.to_string(float_format=lambda x: f"{x:.4f}"))

    add("\n" + "-" * 78)
    add("方案 B · 不合成（帕累托 + 胜负 + 显著性检验）")
    add("-" * 78)
    add("(1) 指标均值")
    add(b["means"].to_string(float_format=lambda x: f"{x:.4f}"))
    add("\n(2) 帕累托前沿（统一为越大越好）——注意：前沿依赖你选哪两个维度")
    for plane, front in b["pareto_by_plane"].items():
        add(f"    {plane:<22} {front}")
    add("\n(3) 逐场景胜负矩阵（行赢列的次数，四指标合计）")
    add(b["win_matrix"].to_string())
    add("\n(4) 显著性检验（α=0.05）")
    for m, st in b["stats"].items():
        add(f"    {METRICS[m][0]:<16} {st['test']:<9} 平均秩={_fmt_ranks(st['avg_rank'])}"
            f"  p={st['p_value']:.4g}  CD={st['cd']:.3f}")
        if st["p_value"] > 0.05:
            add("        ⚠️ 整体检验不显著 → 该指标上所有算法的平均秩差异不可解读")
        for pr in st["significant_pairs"]:
            add(f"        {pr['pair']:<24} |Δ秩|={pr['rank_diff']:.2f}  "
                f"{'显著' if pr['significant'] else '不显著'}")

    k = df.algo.nunique()
    N = df.seed.nunique()
    if k >= 8:
        add(f"\n  ⚠️ k={k} 时 Nemenyi CD = {nemenyi_cd(k, N):.3f}，"
            f"|Δ秩| 需超过该值才算显著；算法越多越难分辨，建议增加场景数 N。")

    add("\n" + "-" * 78)
    add("时间模式敏感性（方案 A 的总分）")
    add("-" * 78)
    tab = {}
    for mode in ("threshold", "ratio", "rank"):
        tab[mode] = score_relative(df, weights, baseline, mode, time_limit_s,
                                   agg=agg)["score"]
    add(pd.DataFrame(tab).to_string(float_format=lambda x: f"{x:.4f}"))
    add("  → 同一份数据，仅因 e 的计分语义不同，排序就可能改变。这就是 D3 必须先拍板的原因。")

    add("\n" + "-" * 78)
    add("聚合方式敏感性（方案 A 的 r_a）")
    add("-" * 78)
    tab2 = {}
    for mode in ("mean", "median", "geomean", "ratio_of_means"):
        tab2[mode] = score_relative(df, weights, baseline, time_mode, time_limit_s,
                                    agg=mode)["r_a"]
    add(pd.DataFrame(tab2).to_string(float_format=lambda x: f"{x:.4f}"))
    add("  → 若四种口径排序不一致，说明结果被少数极端场景驱动，应改用 median/geomean。")

    add("\n" + "-" * 78)
    add("结论")
    add("-" * 78)
    add(f"    方案 A 冠军: {a.index[0]} (score={a.iloc[0]['score']:.4f})")
    add(f"    方案 C 冠军: {c.index[0]} (score={c.iloc[0]['score']:.4f})")
    add(f"    方案 B      : 帕累托前沿 = {b['pareto']}（无单一冠军，各维度各有优势）")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="三套总分口径报告")
    ap.add_argument("--csv", default=None, help="默认 results/exp1/exp1_metrics.csv")
    ap.add_argument("--baseline", default="greedy")
    ap.add_argument("--time-mode", default="threshold", choices=sorted(TIME_MODES))
    ap.add_argument("--time-limit", type=float, default=6.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--agg", default="mean", choices=sorted(AGG_MODES))
    ap.add_argument("--equal-weights", action="store_true")
    args = ap.parse_args()

    from ..config import ROOT
    csv = Path(args.csv) if args.csv else ROOT / "results" / "exp1" / "exp1_metrics.csv"
    df = pd.read_csv(csv)
    weights = EQUAL_WEIGHTS if args.equal_weights else DEFAULT_WEIGHTS
    print(format_report(df, args.baseline, weights, args.time_mode,
                        args.time_limit, args.lam, args.agg))


if __name__ == "__main__":
    main()
