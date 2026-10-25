"""总分接口（scheme A/B/C）的单元测试。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarm_alloc.evaluation.score import (DEFAULT_WEIGHTS, score_pareto,  # noqa: E402
                                          score_relative, score_risk_adjusted,
                                          significance_test)


def _toy(n_scene=10, algos=("greedy", "b_bot", "c_mid")):
    """造一份可控的长表：greedy 最好，c_mid 居中，b_bot 最差。"""
    rows = []
    for s in range(n_scene):
        base = 500 + s
        rows.append(dict(algo="greedy", seed=1000 + s, a_gain=base, b_threat=0.99,
                         d_cost=1.60, time_s=0.01))
        rows.append(dict(algo="c_mid", seed=1000 + s, a_gain=base * 0.99, b_threat=0.90,
                         d_cost=1.70, time_s=0.50))
        rows.append(dict(algo="b_bot", seed=1000 + s, a_gain=base * 0.95, b_threat=0.70,
                         d_cost=1.30, time_s=2.00))
    return pd.DataFrame(rows)[["algo", "seed", "a_gain", "b_threat", "d_cost", "time_s"]]


def test_baseline_scores_exactly_one():
    out = score_relative(_toy(), baseline="greedy")
    assert out.loc["greedy", "score"] == pytest.approx(1.0)
    assert out.loc["greedy", "rank"] == 1


def test_ranking_matches_quality():
    out = score_relative(_toy(), baseline="greedy")
    assert list(out.index) == ["greedy", "c_mid", "b_bot"]


def test_risk_adjusted_with_zero_lambda_equals_relative():
    a = score_relative(_toy())
    c = score_risk_adjusted(_toy(), lam=0.0)
    for algo in a.index:
        assert a.loc[algo, "score"] == pytest.approx(c.loc[algo, "score"])


def test_risk_adjustment_penalises_volatility():
    """给 b_bot 制造高波动，λ 越大它的分越低。"""
    df = _toy()
    mask = (df.algo == "b_bot")
    df.loc[mask, "b_threat"] = np.where(df.loc[mask, "seed"] % 2 == 0, 1.0, 0.3)

    lo = score_risk_adjusted(df, lam=0.0).loc["b_bot", "score"]
    hi = score_risk_adjusted(df, lam=1.0).loc["b_bot", "score"]
    assert hi < lo


def test_weights_are_renormalised():
    w = {"a_gain": 2.0, "b_threat": 2.0, "d_cost": 0.0, "time_s": 0.0}
    out = score_relative(_toy(), weights=w)
    assert out.loc["greedy", "score"] == pytest.approx(1.0)   # 基准恒为 1


def test_unknown_metric_rejected():
    with pytest.raises(ValueError):
        score_relative(_toy(), weights={"nope": 1.0})


def test_pareto_front_finds_tradeoff():
    """c_mid 在 d 上最好、在 b 上居中；greedy 在 b 上最好 → 两者都应在前沿。"""
    out = score_pareto(_toy(), dims=("b_threat", "d_cost"))
    assert "greedy" in out["pareto"]
    assert "c_mid" in out["pareto"]
    assert "b_bot" not in out["pareto"]        # 被两者同时支配


def test_friedman_runs_with_three_algos():
    st = significance_test(_toy(), "b_threat", higher_better=True)
    assert st["test"] == "Friedman"
    assert np.isfinite(st["p_value"])
    assert st["avg_rank"]["greedy"] == pytest.approx(1.0)


def test_wilcoxon_fallback_with_two_algos():
    df = _toy()
    df = df[df.algo != "c_mid"]
    st = significance_test(df, "b_threat", higher_better=True)
    assert st["test"] == "Wilcoxon"
    assert np.isfinite(st["p_value"])


def test_default_weights_sum_to_one():
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------- 真实数据

def test_runs_on_real_exp1_csv():
    from swarm_alloc.config import ROOT
    csv = ROOT / "results" / "exp1" / "exp1_metrics.csv"
    if not csv.exists():
        pytest.skip("exp1_metrics.csv 不存在")
    df = pd.read_csv(csv)
    a = score_relative(df)
    c = score_risk_adjusted(df)
    b = score_pareto(df)
    assert len(a) == df.algo.nunique() == len(c)
    assert abs(a.loc["greedy", "score"] - 1.0) < 1e-9
    assert set(b["pareto"]) <= set(df.algo.unique())
