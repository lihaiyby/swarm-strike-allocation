"""统一模型核心实现（对应《统一模型文档v1》第 2~7 节）。"""
from __future__ import annotations

import numpy as np


class Model:
    def __init__(self, eval_cfg: dict):
        self.kill_threshold = eval_cfg["kill_threshold"]      # 0.9
        self.time_limit_s = eval_cfg["time_limit_s"]          # 1.0
        # ---- D6：距离衰减的毁伤概率（默认关闭，关闭时行为与旧版逐位一致）
        rd = eval_cfg.get("range_decay") or {}
        self.range_decay_on = bool(rd.get("enabled", False))
        self.range_lambda = float(rd.get("lambda", 2.0))
        self.ref_range = float(rd.get("ref_range_m", 3000.0))
        # ---- D6b：硬射程（δ_ij 超过 hard_frac 的机**不可达**，p_ij = 0）
        self.hard_frac = float(rd.get("hard_frac", 0.0))
        # ---- 【批注 B4】资源不足时的优先顺序：value（Long 版）/ threat（旧口径）
        self.select_priority = str(eval_cfg.get("select_priority", "value"))
        # ---- 【批注 B3】我方分组 + 组对组对抗
        gp = eval_cfg.get("grouping") or {}
        self.grouping_on = bool(gp.get("enabled", False))
        self.grouping_mode = str(gp.get("mode", "even"))
        # ---- 【批注 B1】形式 C：距离代价项
        dc = eval_cfg.get("distance_cost") or {}
        self.distance_cost_on = bool(dc.get("enabled", False))
        self.distance_cost_c = float(dc.get("c", 0.0))
        # ---- 【批注 B1】形式 D：时间延迟（命中时刻 = 发射 + d/v）
        td = eval_cfg.get("time_delay") or {}
        self.time_delay_on = bool(td.get("enabled", False))
        self.time_delay_speed = float(td.get("speed_mps", 300.0))

    # ---- D6：距离衰减 ----
    def pair_p(self, p, dist):
        """单架次对单目标的实际毁伤概率（**绝对距离**口径，保留兼容用）。

        ``p_ij = p_j · exp(-λ·d_ij/ref)``
        注意：本项目交战距离约 3000~5000 m，绝对口径下衰减极易把 p 压到接近 0，
        实际仿真请用 :meth:`pair_p_matrix`（相对口径）。
        """
        if not self.range_decay_on:
            return p
        return np.asarray(p) * np.exp(-self.range_lambda * np.asarray(dist) / self.ref_range)

    def pair_p_matrix(self, p_kill, dmat: np.ndarray) -> np.ndarray:
        """(我机 × 目标) 的实际毁伤概率矩阵，**相对距离**口径。

        ``p_ij = p_j · exp(-λ · δ_ij)``，其中
        ``δ_ij = (d_ij - d_j^min) / (d_j^max - d_j^min)`` ∈ [0, 1]

        含义：**对每个目标，最近的那架机不受罚（p 保持原值），越远的机衰减越多。**

        D6b：若 ``hard_frac > 0``，则 ``δ_ij > hard_frac`` 的机**不可达**，``p_ij = 0``。
        这一步把"配错机"从"打得少"升级为"打不到"——即可行性约束而非概率折扣。
        """
        p = np.asarray(p_kill, dtype=float)[None, :]
        if not self.range_decay_on:
            return np.broadcast_to(p, dmat.shape).astype(float).copy()
        dmin = dmat.min(axis=0, keepdims=True)
        dmax = dmat.max(axis=0, keepdims=True)
        span = np.maximum(dmax - dmin, 1.0)          # 防止除零（所有机几乎等距时）
        delta = (dmat - dmin) / span
        out = p * np.exp(-self.range_lambda * delta)
        if self.hard_frac > 0:
            out = np.where(delta > self.hard_frac, 0.0, out)
        return out

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

    # ---- 可行性恢复（文档 §6）：资源不足时放弃低优先级群 ----
    def select_groups(self, groups: list, demand: dict, pool_size: int,
                      values: dict | None = None) -> list:
        """groups: list[Group]（threat 已填）；demand: {group_id: 配额需求}；
        values: {group_id: 群价值}（可选）。

        【批注 B4】优先顺序按 `config/eval.yaml → select_priority` 决定：
        - ``value``（默认，按 `方案04-Long-修改.docx`"优先分配给高价值敌方机组"）
        - ``threat``（旧口径，按熵权 TOPSIS 威胁度）

        返回被保留的群，按优先级从高到低。
        注意：**高价值 ≠ 高威胁**——威胁度还含速度与距离，两者排序可能不同。
        """
        def _val(g):
            return float(values.get(g.id, 0.0)) if values else 0.0

        if self.select_priority == "threat":
            key = lambda g: (-float(g.threat), -_val(g))
        else:
            key = lambda g: (-_val(g), -float(g.threat))

        kept, used = [], 0
        for g in sorted(groups, key=key):
            d = demand.get(g.id, 0)
            if used + d > pool_size:
                continue                      # 跳过该群，继续看优先级更低但更小的群
            kept.append(g)
            used += d
        return kept

    # ---- 惩罚函数（文档 §7，系数待定） ----
    @staticmethod
    def time_penalty(elapsed_s: float, limit_s: float) -> float:
        return max(0.0, elapsed_s - limit_s)
