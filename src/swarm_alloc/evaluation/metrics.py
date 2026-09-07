"""统一评估指标（期望值口径，算法不可见）。

a 打击总收益 / b 威胁消除率 / d 效费比；c 稳定性与 e 求解效率由实验脚本跨场景统计。
"""
from __future__ import annotations

import numpy as np


def per_scene_metrics(X: np.ndarray, state, model, elapsed_s: float) -> dict:
    """对一次分配方案 X 计算单场景指标（期望值口径，不掷骰子）。"""
    values, p_kill, _ = state.enemy_matrix()
    counts = X.sum(axis=0)
    P = 1.0 - (1.0 - p_kill) ** counts

    a = float(np.dot(values, P))

    killed = P >= model.kill_threshold            # 歼灭判定：P_j ≥ 0.9
    threat_all = sum(g.threat for g in state.groups) + 1e-12
    threat_dead = 0.0
    for g in state.groups:
        js = [i for i, eid in enumerate(state.enemy_ids) if eid in g.member_ids]
        if js and all(killed[j] for j in js):
            threat_dead += g.threat
    b = threat_dead / threat_all

    used = int(X.sum())
    d = a / max(used, 1)                          # 单机价值 1

    return {"a_gain": a, "b_threat": b, "d_cost": d,
            "used": used, "time_s": elapsed_s,
            "n_enemy": len(state.enemy_ids), "n_pool": len(state.pool_ids)}
