"""config 读取：全工程唯一参数来源，代码中禁止硬编码。"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录 swarm-strike-allocation/
CONFIG_DIR = ROOT / "config"


def load(name: str) -> dict:
    """读取 config/<name>.yaml，name ∈ scenario / enemy_types / friendly / eval"""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
