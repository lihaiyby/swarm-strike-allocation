"""指标 b（威胁消除率）的**唯一实现**。

设计依据见 `docs/擂台改造-任务清单.md` D2 = ②：
    b = Σ_{k: 全部成员被歼} V_k  /  Σ_k V_k,   V_k = Σ_{j∈k} v_j

两个关键性质（这正是旧实现缺失的）：
1. **分母与算法无关**：若 `groups` 是场景的真值团划分，则对方差无贡献的
   分母恒等于场景总价值 Σ_j v_j——任何算法、任何回合数都得到同一个数。
2. **群划分固定**：不做动态重聚类，因此不会出现"幸存敌机被反复计入分母"的漂移。

旧实现（simulate.py 的 `seen_groups`/`threat_total`）与本模块的差异见
`docs/擂台改造-进度记录.md` 2026-10-23 条目：实测同一场景 greedy 与 ga_iap 的
累计群 ID 数分别为 21 与 53（真实团数 21），分母膨胀 2.52 倍。
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def group_elimination_rate(
    P: Mapping[int, float],
    groups: Sequence[Sequence[int]],
    values: Mapping[int, float],
    threshold: float = 0.9,
) -> tuple:
    """计算威胁消除率 b。

    参数
    ----
    P : {enemy_id: 累积毁伤概率}。缺失的敌机按毁伤 0 处理（未接战 = 未歼灭）。
    groups : 群划分，list[list[enemy_id]]；应为对全部敌机的**划分**。
    values : {enemy_id: 价值}。
    threshold : 歼灭阈值，默认 0.9（= eval.yaml 的 kill_threshold）。

    返回
    ----
    (b, numerator, denominator)
        b = numerator / denominator；denominator 为 0 时返回 0.0。

    注：判据是"群内**全部**成员 P_j ≥ threshold"（一票否决），
    与《统一模型文档v1》§8 一致。侦察机不设豁免。
    """
    num = 0.0
    den = 0.0
    for members in groups:
        group_value = 0.0
        all_dead = True
        for eid in members:
            v = float(values[eid])
            group_value += v
            if P.get(eid, 0.0) < threshold:
                all_dead = False
        den += group_value
        if all_dead:
            num += group_value
    if den <= 0.0:
        return 0.0, 0.0, 0.0
    return num / den, num, den


def total_value(values: Iterable[float]) -> float:
    """场景总价值——真值团口径下 b 的分母（常数，与算法无关）。"""
    return float(sum(values))
