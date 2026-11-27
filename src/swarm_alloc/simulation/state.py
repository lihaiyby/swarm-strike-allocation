"""态势数据结构：算法接口的统一输入。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class State:
    t: float
    enemies: dict            # enemy_id -> EnemyUnit（当前在空，pos 已推进到时刻 t）
    groups: list             # list[Group]，threat 已由 TOPSIS 填好
    pool: dict               # friendly_id -> FriendlyUnit（可用）
    frozen_ids: list = field(default_factory=list)   # 已下达指令，不可改派
    # ---- 【批注 B3】我方分组与配对（仅在 grouping 启用时非空）
    friendly_groups: dict = field(default_factory=dict)   # fid -> 我方组号
    group_match: dict = field(default_factory=dict)       # 我方组号 -> 敌群 id
    # ---- 【V2-P2-3】在途打击（时序口径才非空；v1 固定时刻口径恒为空）
    # 语义：已经派出去、还在飞、尚未结算的打击。算法可据此避免重复派兵，
    # 也可把"已在途的期望毁伤"计入边际收益。
    inflight_count: dict = field(default_factory=dict)    # eid -> 在途架数
    inflight_p: dict = field(default_factory=dict)        # eid -> 在途累积期望毁伤 1-Π(1-p)
    inflight_arrive: dict = field(default_factory=dict)   # eid -> list[到达时刻]（按升序）

    @property
    def inflight_ids(self) -> list:
        """所有在途打击的目标 id（有在途的目标）。"""
        return sorted(self.inflight_count.keys())

    @property
    def enemy_ids(self) -> list:
        return sorted(self.enemies.keys())

    @property
    def pool_ids(self) -> list:
        return sorted(self.pool.keys())

    # ---- 【V2-P6】紧迫性加权（老师建议 8：分配要考虑速度与方向，不能只看价值）
    urgency_beta: float = 0.0      # >0 时按"径向接近速率"放大目标在算法眼中的价值
    defense_radius: float = 700.0  # 仅用于计算 ETA（保留给未来扩展）

    def _urgency(self) -> np.ndarray:
        """每个目标的紧迫性 ∈ [0,1]：径向接近速率（含方向）在当前目标间的归一化。

        朝基地飞得越快 → 越紧迫。全体同向时归一化后无差异，返回全 0.5（中性）。
        该值**只影响算法的决策价值**，不影响评分（评分仍用真实 `v_j`）。
        """
        eids = self.enemy_ids
        n = len(eids)
        if n == 0:
            return np.zeros(0)
        vr = np.zeros(n)
        for j, eid in enumerate(eids):
            u = self.enemies[eid]
            r = np.asarray(u.pos, dtype=float)
            rn = float(np.linalg.norm(r))
            if rn > 1e-9:
                vr[j] = -float(np.dot(np.asarray(u.vel, dtype=float), r / rn))
        lo, hi = float(vr.min()), float(vr.max())
        if hi - lo < 1e-9:
            return np.full(n, 0.5)
        return (vr - lo) / (hi - lo)

    def enemy_matrix(self) -> tuple:
        """返回 (values, p_kill, positions)，按 enemy_ids 顺序。

        【V2-P6】``urgency_beta > 0`` 时，``values`` 变为"紧迫性加权后的决策价值"：
        ``v_eff = v · (1 + β·urgency)``；``urgency_beta = 0``（默认）时与旧版逐位一致。
        """
        us = [self.enemies[i] for i in self.enemy_ids]
        vals = np.array([u.value for u in us], dtype=float)
        if self.urgency_beta > 0 and len(vals):
            vals = vals * (1.0 + self.urgency_beta * self._urgency())
        return (vals,
                np.array([u.p_kill for u in us], dtype=float),
                np.vstack([u.pos for u in us]))

    def pool_matrix(self) -> np.ndarray:
        return np.vstack([self.pool[i].pos for i in self.pool_ids])

    # ---- 【V2-P2-3】在途可见性 ----
    def inflight_count_of(self, eid) -> int:
        """该目标当前在途（已派出、未结算）的打击架数。"""
        return int(self.inflight_count.get(eid, 0))

    def inflight_p_of(self, eid) -> float:
        """该目标在途打击的**累积期望毁伤** `1 - Π(1 - p_ij)`（不含已结算部分）。"""
        return float(self.inflight_p.get(eid, 0.0))

    def remaining_quota(self, model, p_kill_of) -> np.ndarray:
        """每个目标**还差几架**才满足配额（在途架数已计入）。

        返回与 ``enemy_ids`` 同序的 int 数组；已饱和的目标为 0。
        算法可直接用它来裁剪派兵数量，避免重复投入。
        """
        eids = self.enemy_ids
        out = np.zeros(len(eids), dtype=int)
        for j, eid in enumerate(eids):
            m = int(model.quota(p_kill_of(eid), model.kill_threshold))
            out[j] = max(0, m - self.inflight_count_of(eid))
        return out

    def effective_p0(self) -> np.ndarray:
        """每个目标的"起始累积毁伤"= 在途部分（用于边际收益计算）。

        算法若把边际收益写成 "从 p0 出发再加 k 架"，就自动考虑了在途打击。
        """
        return np.array([self.inflight_p_of(e) for e in self.enemy_ids], dtype=float)

    # ---- D6 ----
    def pair_p_matrix(self, model) -> np.ndarray:
        """(我机 × 目标) 的实际单架次毁伤概率矩阵 ``p_ij``（相对距离口径）。

        关闭距离衰减时退化为"每目标一个常数"的广播结果，与旧口径一致。
        算法可用它来区分"近机"与"远机"——这是 D6 引入的唯一新信息。
        """
        _, p, epos = self.enemy_matrix()
        fpos = self.pool_matrix()
        d = np.linalg.norm(fpos[:, None, :] - epos[None, :, :], axis=2)
        return model.pair_p_matrix(p, d)

    # ---- 【批注 B3】C3 允许矩阵 ----
    def c3_allowed(self) -> np.ndarray:
        """(我机 × 目标) 的布尔矩阵：True 表示该机允许打该目标。

        未启用分组时全为 True（等价于无 C3）。
        """
        n_f, n_e = len(self.pool_ids), len(self.enemy_ids)
        if not self.friendly_groups or not self.group_match:
            return np.ones((n_f, n_e), dtype=bool)
        enemy_group_of = {u: g.id for g in self.groups for u in g.member_ids}
        out = np.zeros((n_f, n_e), dtype=bool)
        for i, fid in enumerate(self.pool_ids):
            tgt = self.group_match.get(self.friendly_groups.get(fid))
            if tgt is None:
                continue
            for j, eid in enumerate(self.enemy_ids):
                if enemy_group_of.get(eid) == tgt:
                    out[i, j] = True
        return out

