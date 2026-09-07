"""波次结算（方案04-v3：战损与波次同步结算）。

口径：
- 毁伤判定用期望值口径——累积毁伤 P_j ≥ 0.9 即判定摧毁，不掷骰子（指标可复现）；
- 战损抽样用场景种子驱动的 rng（逐次运行可复现）：
  格斗型被接战时以概率 q1 额外击落资源池中离它最近的 1 架；
  范围伤害型被接战时，资源池中距它 r 以内的无人机各以概率 q2 损失。
"""
from __future__ import annotations

import numpy as np


def battle_damage(rng: np.random.Generator, engaged_enemies: list,
                  pool: dict, type_lethality: dict) -> list:
    """对本轮回合被接战的敌机结算我方战损。返回损失的我方无人机 id 列表。"""
    lost = []
    if not pool:
        return lost
    pool_ids = sorted(pool.keys())
    pool_pos = {i: pool[i].pos for i in pool_ids}

    for e in engaged_enemies:
        leth = type_lethality.get(e.type_name)
        if not leth:
            continue
        if leth["mode"] == "point":
            if rng.random() < leth["q"]:
                d = {i: np.linalg.norm(p - e.pos) for i, p in pool_pos.items()
                     if i not in lost}
                if d:
                    lost.append(min(d, key=d.get))
        elif leth["mode"] == "area":
            for i in pool_ids:
                if i in lost:
                    continue
                if np.linalg.norm(pool_pos[i] - e.pos) <= leth["radius_m"] \
                        and rng.random() < leth["q"]:
                    lost.append(i)
    return lost
