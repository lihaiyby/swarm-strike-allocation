"""算法接口层：抽象基类 + 注册表。"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class AlgoBase(ABC):
    """所有候选算法的统一接口。

    solve(state, model) -> X
    - 只读 state（态势）与 model（统一模型），禁止内部改写毁伤/配额/约束；
    - 返回 X：(len(state.pool_ids) × len(state.enemy_ids)) 的 0-1 矩阵，
      X[i,j]=1 表示 pool_ids[i] 打击 enemy_ids[j]；
    - 框架计时并计入评估指标 e。
    """

    name: str = "base"

    @abstractmethod
    def solve(self, state, model) -> np.ndarray:
        ...


_REGISTRY: dict = {}


def register(cls):
    _REGISTRY[cls.name] = cls
    return cls


def get(name: str):
    return _REGISTRY[name]()


def available() -> list:
    return sorted(_REGISTRY)
