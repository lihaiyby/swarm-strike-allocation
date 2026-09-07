"""单波次敌机生成：高斯团 + 散点（肉眼不可分、密度可分）。"""
from __future__ import annotations

import numpy as np

from .scene import EnemyUnit, Wave


def sample_hemisphere(rng: np.random.Generator, R: float, n: int = 1) -> np.ndarray:
    """上半球面均匀采样，返回 (n,3)，|pos| = R。"""
    v = rng.normal(size=(n, 3))
    v[:, 2] = np.abs(v[:, 2])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * R


def _pick_centers(rng, R, k, sigma, min_sep_sigma):
    """在外球面选 k 个团心，两两弦距 ≥ min_sep_sigma * σ（拒绝采样）。"""
    min_chord = min_sep_sigma * sigma
    centers, tries = [], 0
    while len(centers) < k and tries < 20000:
        c = sample_hemisphere(rng, R, 1)[0]
        if all(np.linalg.norm(c - o) >= min_chord for o in centers):
            centers.append(c)
        tries += 1
    if len(centers) < k:
        raise RuntimeError(f"团心拒绝采样失败：{len(centers)}/{k}，请减小 min_center_sep_sigma")
    return centers


def _group_sizes(rng, n_cluster: int, k: int, jitter: float) -> list:
    """每群数量 = n_cluster/k ± jitter 随机扰动，修正使总和 = n_cluster。"""
    base = n_cluster / k
    sizes = [max(1, round(base * rng.uniform(1 - jitter, 1 + jitter))) for _ in range(k)]
    diff = n_cluster - sum(sizes)
    i = int(np.argmax(sizes))
    sizes[i] += diff  # 余数归入最大群
    assert all(s >= 1 for s in sizes), f"群数量非法: {sizes}"
    return sizes


def _assign_types(rng, n: int, type_cfgs: list) -> list:
    """按类型占比抽样，返回长度 n 的类型配置列表。"""
    ratios = np.array([t["ratio"] for t in type_cfgs])
    ratios = ratios / ratios.sum()
    idx = rng.choice(len(type_cfgs), size=n, p=ratios)
    return [type_cfgs[i] for i in idx]


def gen_wave(rng: np.random.Generator, t: float, n_wave: int, R: float,
             scfg: dict, type_cfgs: list, id_start: int) -> tuple[Wave, int]:
    """生成一个波次。返回 (Wave, 下一个可用 id)。"""
    cl = scfg["cluster"]
    sigma = rng.uniform(*cl["sigma_m"])
    # 二项随机取噪点：期望 = n_wave×noise_ratio，避免 round() 导致噪点数恒定
    n_noise = int(rng.binomial(n_wave, cl["noise_ratio"]))
    n_cluster = n_wave - n_noise
    k = scfg["waves"]["groups_per_wave"]

    centers = _pick_centers(rng, R, k, sigma, cl["min_center_sep_sigma"])
    sizes = _group_sizes(rng, n_cluster, k, cl["size_jitter"])

    # 位置：团内 3D 高斯偏移后重投影到外球面
    positions, clusters = [], []
    for ci, (c, s) in enumerate(zip(centers, sizes)):
        pts = c + rng.normal(0, sigma, size=(s, 3))
        pts[:, 2] = np.abs(pts[:, 2])          # 保持上半球
        pts *= R / np.linalg.norm(pts, axis=1, keepdims=True)
        positions.append(pts)
        clusters.extend([ci] * s)
    if n_noise > 0:
        positions.append(sample_hemisphere(rng, R, n_noise))
        clusters.extend([-1] * n_noise)
    positions = np.vstack(positions)

    types = _assign_types(rng, n_wave, type_cfgs)
    units = []
    for i in range(n_wave):
        tc = types[i]
        speed = rng.uniform(*tc["speed_mps"]) if tc["speed_mps"][0] < tc["speed_mps"][1] \
            else float(tc["speed_mps"][0])
        pos = positions[i]
        units.append(EnemyUnit(
            id=id_start + i, pos=pos, vel=-speed * pos / R, speed=speed,
            type_name=tc["name"], value=tc["value"], maneuver=tc["maneuver"],
            p_kill=tc["p_kill"], is_noise=(clusters[i] == -1), cluster=clusters[i],
        ))
    return Wave(t=t, units=units), id_start + n_wave
