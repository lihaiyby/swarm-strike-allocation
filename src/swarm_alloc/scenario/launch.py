"""我方部署：环形发射平台 + 垂直升空（V2-P1-1 / P1-2）。

老师建议（2026-11-17，第 3 条）
------------------------------
> 我方无人机不是在内球面均匀随机分布，而是在内球面在地面形成的**圆形投影**上
> 呈某种规律分布的若干个点，每个点视作一个**无人机发射平台**，可以发射 4 架（或几架）；
> 无人机发射后**升到一定高度**后直线朝着目标前进，不考虑发射耗时，考虑**直线升空**的耗时，
> 直线升空按 **30 米**算。

老师的决策（2026-11-22）
------------------------
- 布局：**环形均匀分布**
- 垂直升空速度：**20 m/s** → 30 m ÷ 20 m/s = **1.5 s**
- 巡航速度：**70 m/s 为最大值**

实现约定
--------
1. 平台位于地面（z=0），分布在以基地为圆心、半径落在 ``[ring_inner_frac, ring_outer_frac] × 防线半径``
   的若干同心环上，每环平台**角度均布**、相邻环**错开半个扇区**（避免正对同一方向）；
2. 我方无人机起飞后先垂直升空 ``climb_height_m``，**起点坐标 = 平台上方 climb_height_m**；
3. 每架机的巡航速度在 ``speed_mps`` 区间内独立抽样（保持"同质"指型号相同，速度可有个体差异）；
4. 升空耗时 ``t_climb = climb_height_m / climb_speed_mps``，单独返回，便于分开统计。
"""
from __future__ import annotations

import numpy as np


def build_launch_sites(rng, defense_r: float, n_drones: int, cfg: dict) -> np.ndarray:
    """生成发射平台的平面坐标，返回 (n_platforms, 2) 数组。"""
    per = max(1, int(cfg.get("drones_per_platform", 4)))
    n_plat_cfg = cfg.get("n_platforms")
    n_plat = int(n_plat_cfg) if n_plat_cfg else int(np.ceil(n_drones / per))
    n_plat = max(1, n_plat)

    rings = max(1, int(cfg.get("rings", 2)))
    r_in = float(cfg.get("ring_inner_frac", 0.35)) * defense_r
    r_out = float(cfg.get("ring_outer_frac", 0.85)) * defense_r
    radii = np.linspace(r_in, r_out, rings) if rings > 1 else np.array([0.5 * (r_in + r_out)])

    # 每环平台数：均分，余数从前面的环补齐
    per_ring = np.full(rings, n_plat // rings, dtype=int)
    for i in range(n_plat - int(per_ring.sum())):
        per_ring[i % rings] += 1

    jitter = float(cfg.get("angle_jitter_rad", 0.0))
    base_offset = float(cfg.get("angle_offset_rad", 0.0))

    sites = []
    for k, (r, m) in enumerate(zip(radii, per_ring)):
        if m <= 0:
            continue
        # 相邻环错开半个扇区
        offset = base_offset + (k * np.pi / m if m else 0.0)
        ang = offset + np.arange(m) * (2.0 * np.pi / m)
        if jitter > 0:
            ang = ang + rng.uniform(-jitter, jitter, size=m)
        sites.append(np.column_stack([r * np.cos(ang), r * np.sin(ang)]))
    return np.vstack(sites)


def deploy_ring_platform(rng, defense_r: float, n_drones: int, cfg: dict) -> dict:
    """按"环形平台 + 升空"部署我方无人机。

    返回 ``dict``：
    - ``pos``      (n, 3) 我方**升空后**的起点坐标
    - ``sites``    (p, 2) 平台平面坐标（诊断/绘图用）
    - ``site_idx`` (n,)   每架机所属平台编号
    - ``t_climb``  float  垂直升空耗时（秒）
    """
    sites = build_launch_sites(rng, defense_r, n_drones, cfg)
    site_idx = np.arange(n_drones) % len(sites)
    xy = sites[site_idx]
    z = np.full(n_drones, float(cfg.get("climb_height_m", 30.0)))
    t_climb = float(cfg.get("climb_height_m", 30.0)) / max(
        float(cfg.get("climb_speed_mps", 20.0)), 1e-9)
    return dict(pos=np.column_stack([xy, z]), sites=sites,
                site_idx=site_idx, t_climb=t_climb)


def flight_time(d_climb: float, climb_h: float, climb_v: float,
                d_horiz: float, cruise_v: float) -> float:
    """一次打击的飞行时间 = 升空耗时 + 水平飞行耗时（秒）。

    当前简化：``d_climb`` 视作已到升空高度（默认取 ``climb_h``），
    所以第一项恒为 ``climb_h / climb_v``。保留该参数便于后续区分"已在空中"的机。
    """
    t_up = max(climb_h - d_climb, 0.0) / max(climb_v, 1e-9)
    return t_up + d_horiz / max(cruise_v, 1e-9)
