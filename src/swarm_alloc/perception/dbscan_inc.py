"""增量 DBSCAN 分组。

规则（方案04-v3）：
- 每次重分配时对当前在空敌机运行 DBSCAN；
- 已存在的群保留原 ID：新簇与旧群按成员重叠数最大匹配；
- 噪点（分散无人机）独立成群：每个噪点自成一个单目标群。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN


@dataclass
class Group:
    id: int
    member_ids: list       # 敌机 id 列表
    threat: float = 0.0    # 威胁度（由 TOPSIS 填入）


@dataclass
class GroupTracker:
    """跨触发点保持群 ID 稳定。"""
    eps: float
    min_samples: int
    groups: dict = field(default_factory=dict)   # group_id -> Group
    _next_id: int = 0

    def update(self, enemy_ids: list, positions: np.ndarray) -> dict:
        """输入当前在空敌机 id 与位置 (n,3)，返回 {enemy_id: group_id}。"""
        n = len(enemy_ids)
        if n == 0:
            self.groups = {}
            return {}

        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(positions)
        # 簇 label → 成员；噪点 label=-1 独立成群
        raw = {}
        for i, lab in enumerate(labels):
            raw.setdefault(int(lab), []).append(enemy_ids[i])

        new_groups, used_old = {}, set()
        # 先处理成簇（label >= 0）：与旧群按成员重叠最大匹配，保留旧 ID
        for lab, members in sorted((k, v) for k, v in raw.items() if k >= 0):
            best, best_overlap = None, 0
            mset = set(members)
            for gid, g in self.groups.items():
                if gid in used_old:
                    continue
                ov = len(mset & set(g.member_ids))
                if ov > best_overlap:
                    best, best_overlap = gid, ov
            gid = best if best is not None else self._fresh_id()
            used_old.add(gid)
            new_groups[gid] = Group(id=gid, member_ids=members)
        # 噪点：每点独立成群（新 ID）
        for member in raw.get(-1, []):
            gid = self._fresh_id()
            new_groups[gid] = Group(id=gid, member_ids=[member])

        self.groups = new_groups
        return {uid: g.id for g in new_groups.values() for uid in g.member_ids}

    def _fresh_id(self) -> int:
        while self._next_id in self.groups:
            self._next_id += 1
        gid = self._next_id
        self._next_id += 1
        return gid
