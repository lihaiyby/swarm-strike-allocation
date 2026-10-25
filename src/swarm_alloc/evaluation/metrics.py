"""统一评估指标（期望值口径，算法不可见）。

a 打击总收益 / b 威胁消除率 / d 效费比；c 稳定性与 e 求解效率由实验脚本跨场景统计。

⚠️ 关于 b：本模块是**单轮诊断**口径（只见当前态势），与仿真主循环的全程口径不同。
   权威数值是 `simulation.simulate.BattleResult.b_threat`（真值团 + 价值加权 + 全程一次计算）。
   两处共用 `evaluation/b_metric.group_elimination_rate` 这一份公式实现（任务 P0.6）。
"""
from __future__ import annotations

import numpy as np

from .b_metric import group_elimination_rate


def per_scene_metrics(X: np.ndarray, state, model, elapsed_s: float,
                      groups=None) -> dict:
    """对一次分配方案 X 计算单场景指标（期望值口径，不掷骰子）。

    groups : 可选，list[list[enemy_id]] 群划分。缺省用感知层的动态分组 `state.groups`
             （单轮诊断）。要评估真值团口径请显式传入。
    """
    values_vec, p_kill, _ = state.enemy_matrix()
    counts = X.sum(axis=0)
    P_vec = 1.0 - (1.0 - p_kill) ** counts

    a = float(np.dot(values_vec, P_vec))

    eids = state.enemy_ids
    P = {eid: float(p) for eid, p in zip(eids, P_vec)}
    values = {eid: float(v) for eid, v in zip(eids, values_vec)}
    if groups is None:
        groups = [list(g.member_ids) for g in state.groups]
    b, _, _ = group_elimination_rate(P, groups, values, model.kill_threshold)

    used = int(X.sum())
    d = a / max(used, 1)                          # 单机价值 1

    return {"a_gain": a, "b_threat": b, "d_cost": d,
            "used": used, "time_s": elapsed_s,
            "n_enemy": len(eids), "n_pool": len(state.pool_ids)}
