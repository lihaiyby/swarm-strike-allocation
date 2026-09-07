"""算法包：自动发现并导入本目录下所有算法模块，触发 @register 注册。

新增算法只需在本目录新建 <name>.py 并用 @register 装饰类，
无需改动任何实验脚本的 import。
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

for _m in pkgutil.iter_modules([str(Path(__file__).parent)]):
    if _m.name != "base":
        importlib.import_module(f".{_m.name}", __package__)
