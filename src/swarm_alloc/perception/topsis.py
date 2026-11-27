"""熵权法-TOPSIS 群威胁评估。

【V2-P6】老师建议 2026-11-17 第 8 条：
> "每次分配都要考虑敌机价值，速度，**速度方向**等等参数，也就算是综合考虑，不能只看价值"

因此把威胁评估从"价值 / 速度 / 距离"扩展为可配置的**特征集**，新增两个把
"速度 + 方向 + 位置"合成一个物理量的特征：

| 特征 | 定义 | 方向 |
|---|---|---|
| `value` | 群总价值 Σv | 越大越威胁 |
| `speed` | 群平均速度 | 越大越威胁 |
| `dist` | 群到基地最近距离 | 越小越威胁 |
| **`closure`** | **平均径向接近速率** $\\overline{-\\mathbf{v}_i\\cdot\\hat{\\mathbf{r}}_i}$（含**方向**） | 越大越威胁 |
| **`eta`** | **最早突防剩余时间** $\\min_i (\\lVert\\mathbf{r}_i\\rVert-R_d)/v_{r,i}$ | **越小越威胁** |

`closure` 与 `eta` 都同时用到速度大小与方向：敌机若背向基地飞，`closure` 为负、
`eta` 为正无穷（不紧迫）。

配置（`config/scenario.yaml → perception.threat`）：

```yaml
perception:
  threat:
    features: [value, closure, eta]     # 旧口径为 [value, speed, dist]
    defense_radius_m: 700               # 仅 eta 使用；缺省读 battlefield
```

熵权法自动给"无区分度"的特征降到 0 权重，所以多放几个特征不会破坏结果。
"""
from __future__ import annotations

import numpy as np

from .dbscan_inc import Group

# 支持的特征及其"正向化"方式（True=越大越威胁）
FEATURES = {
    "value": True,
    "speed": True,
    "dist": False,
    "closure": True,
    "eta": False,
}
DEFAULT_FEATURES = ["value", "speed", "dist"]        # 与 v1 完全一致


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


def _radial(us, r_min: float) -> tuple:
    """返回 (平均径向接近速率, 最早突防剩余时间)。

    径向速度 $v_r = \\mathbf{v}\\cdot\\hat{\\mathbf{r}}$；朝基地飞时 $v_r<0$，
    故"接近速率"取 $-v_r$。$\\hat{\\mathbf{r}}$ 用单位向量，避免距离量纲影响。
    """
    clos, etas = [], []
    for u in us:
        r = np.asarray(u.pos, dtype=float)
        rn = float(np.linalg.norm(r))
        if rn < 1e-9:
            clos.append(0.0)
            etas.append(0.0)
            continue
        vr = float(np.dot(np.asarray(u.vel, dtype=float), r / rn))
        clos.append(-vr)
        if vr < 0:                            # 正在接近
            etas.append(max(rn - r_min, 0.0) / (-vr))
        else:                                 # 不接近 → 不紧迫
            etas.append(np.inf)
    return float(np.mean(clos)), float(min(etas))


def _feature_matrix(groups: list, units: dict, features: list,
                    defense_radius: float) -> np.ndarray:
    rows = []
    for g in groups:
        us = [units[i] for i in g.member_ids]
        vals = {
            "value": float(sum(u.value for u in us)),
            "speed": float(np.mean([u.speed for u in us])),
            "dist": float(min(np.linalg.norm(u.pos) for u in us)),
        }
        if "closure" in features or "eta" in features:
            clos, eta = _radial(us, defense_radius)
            vals["closure"] = clos
            vals["eta"] = eta
        rows.append([vals[f] for f in features])
    return np.array(rows, dtype=float)


def features_from_config(scfg: dict) -> tuple:
    """从 `scenario.yaml → perception.threat` 读 (features, defense_radius)。

    缺省返回**旧口径** `["value","speed","dist"]` 与 `battlefield.defense_radius_m`，
    保证没有配置时行为与 v1 完全一致。
    """
    perc = (scfg or {}).get("perception") or {}
    th = perc.get("threat") or {}
    feats = th.get("features") or DEFAULT_FEATURES
    r = th.get("defense_radius_m")
    if r is None:
        r = ((scfg or {}).get("battlefield") or {}).get("defense_radius_m", 700.0)
    return list(feats), float(r)


def topsis(groups: list, units: dict, features: list | None = None,
           defense_radius: float = 700.0) -> list:
    """groups: list[Group]；units: {enemy_id: EnemyUnit}。回填 threat 并返回 groups。

    ``features`` 缺省为 ``DEFAULT_FEATURES``（与 v1 逐位一致）；
    传入 ``["value","closure","eta"]`` 即启用"速度+方向+紧迫性"口径。
    """
    if not groups:
        return groups
    feats = list(features or DEFAULT_FEATURES)
    unknown = [f for f in feats if f not in FEATURES]
    if unknown:
        raise ValueError(f"未知威胁特征 {unknown}；可选 {sorted(FEATURES)}")

    M = _feature_matrix(groups, units, feats, defense_radius)
    # 正向化：越大越威胁的保持原值，越小越威胁的取负
    M_pos = np.column_stack([M[:, j] if FEATURES[f] else -M[:, j]
                             for j, f in enumerate(feats)])
    # 无穷大（永不突防）用该列最大值替代，避免熵权法出 NaN
    if not np.isfinite(M_pos).all():
        col_max = np.nanmax(np.where(np.isfinite(M_pos), M_pos, np.nan), axis=0)
        M_pos = np.where(np.isfinite(M_pos), M_pos, col_max)

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
