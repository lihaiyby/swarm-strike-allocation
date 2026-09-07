"""熵权法-TOPSIS 群威胁评估。

评估依据（方案04-v3）：群总价值、群平均速度、群距基地最近距离。
熵权法自动赋权，TOPSIS 逼近理想解排序，输出各群威胁度 w_k ∈ [0,1]（越大越危险）。
"""
from __future__ import annotations

import numpy as np

from .dbscan_inc import Group


def entropy_weights(M: np.ndarray) -> np.ndarray:
    """熵权法：输入 (m 群 × n 指标) 决策矩阵（已正向化），输出权重。"""
    m = M.shape[0]
    if m <= 1:                               # 单群时无区分度可言，等权
        return np.full(M.shape[1], 1.0 / M.shape[1])
    P = M / (M.sum(axis=0, keepdims=True) + 1e-12)
    e = -np.nansum(P * np.log(P + 1e-12), axis=0) / np.log(m)
    d = 1 - e
    if not np.isfinite(d).all() or d.sum() < 1e-12:   # 所有指标无区分度
        return np.full(M.shape[1], 1.0 / M.shape[1])
    return d / d.sum()


def topsis(groups: list, units: dict) -> list:
    """groups: list[Group]；units: {enemy_id: EnemyUnit}。回填 threat 并返回 groups。"""
    if not groups:
        return groups
    feats = []
    for g in groups:
        us = [units[i] for i in g.member_ids]
        total_value = sum(u.value for u in us)
        mean_speed = float(np.mean([u.speed for u in us]))
        min_dist = min(float(np.linalg.norm(u.pos)) for u in us)
        feats.append([total_value, mean_speed, min_dist])
    M = np.array(feats, dtype=float)

    # 正向化：价值、速度越大越危险；距离越小越危险（取负）
    M_pos = np.column_stack([M[:, 0], M[:, 1], -M[:, 2]])
    w = entropy_weights(M_pos)

    # TOPSIS：向量归一化 → 加权 → 正负理想解距离
    V = M_pos / (np.linalg.norm(M_pos, axis=0, keepdims=True) + 1e-12) * w
    v_best, v_worst = V.max(axis=0), V.min(axis=0)
    d_best = np.linalg.norm(V - v_best, axis=1)
    d_worst = np.linalg.norm(V - v_worst, axis=1)
    score = d_worst / (d_best + d_worst + 1e-12)

    for g, s in zip(groups, score):
        g.threat = float(s)
    return groups
