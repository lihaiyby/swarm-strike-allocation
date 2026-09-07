"""统一模型核心实现（对应《统一模型文档v1》第 2~7 节）。"""
from __future__ import annotations

import numpy as np


class Model:
    def __init__(self, eval_cfg: dict):
        self.kill_threshold = eval_cfg["kill_threshold"]      # 0.9
        self.time_limit_s = eval_cfg["time_limit_s"]          # 1.0

    # ---- 毁伤与配额（文档 §2 §5） ----
    @staticmethod
    def damage(p: float, k: int) -> float:
        """累积毁伤概率 P(k) = 1 - (1-p)^k。"""
        return 1.0 - (1.0 - p) ** k

    @staticmethod
    def quota(p: float, threshold: float = 0.9) -> int:
        """达到期望毁伤 threshold 的最小架数 m = ceil(ln(1-threshold)/ln(1-p))。"""
        if p >= 1.0:
            return 1
        return int(np.ceil(np.log(1 - threshold) / np.log(1 - p)))

    # ---- 目标函数（文档 §4，= 指标 a） ----
    def objective(self, X: np.ndarray, values: np.ndarray, p_kill: np.ndarray) -> float:
        """X: (m_pool × n_enemy) 0-1 矩阵；返回打击总收益 Σ v_j·P_j。"""
        counts = X.sum(axis=0)                                # 每目标分配架数
        P = 1.0 - (1.0 - p_kill) ** counts
        return float(np.dot(values, P))

    # ---- 约束检查（文档 §5 C1~C5），返回违规信息列表 ----
    def check(self, X: np.ndarray, state, quotas: np.ndarray,
              group_of_enemy: dict, group_of_friendly: dict) -> list:
        errs = []
        row_sums = X.sum(axis=1)
        if (row_sums > 1).any():
            errs.append(f"C1 单架单目标: {int((row_sums > 1).sum())} 架被重复分配")
        col_sums = X.sum(axis=0)
        for j, c in enumerate(col_sums):
            if 0 < c < quotas[j]:
                errs.append(f"C2 配额下限: 目标 {state.enemy_ids[j]} 分配 {int(c)} < {int(quotas[j])} 架")
        for i, fid in enumerate(state.pool_ids):
            gk = group_of_friendly.get(fid)
            for j, eid in enumerate(state.enemy_ids):
                if X[i, j] and gk is not None and group_of_enemy.get(eid) != gk:
                    errs.append(f"C3 群对群: 编组 {gk} 的 {fid} 分配到了群外目标 {eid}")
                    break
        frozen = set(state.frozen_ids)
        for i, fid in enumerate(state.pool_ids):
            if fid in frozen and row_sums[i] > 0:
                errs.append(f"C5 冻结机被分配: {fid}")
        return errs

    # ---- 可行性恢复（文档 §6）：按威胁度纳入，放弃低威胁群 ----
    @staticmethod
    def select_groups(groups: list, demand: dict, pool_size: int) -> list:
        """groups: list[Group]（threat 已填）；demand: {group_id: 配额需求}。
        返回被保留的群（按威胁度从高到低）。"""
        kept, used = [], 0
        for g in sorted(groups, key=lambda g: -g.threat):
            d = demand.get(g.id, 0)
            if used + d > pool_size:
                continue                      # 跳过该群，继续看更低威胁但更小的群
            kept.append(g)
            used += d
        return kept

    # ---- 惩罚函数（文档 §7，系数待定） ----
    @staticmethod
    def time_penalty(elapsed_s: float, limit_s: float) -> float:
        return max(0.0, elapsed_s - limit_s)
