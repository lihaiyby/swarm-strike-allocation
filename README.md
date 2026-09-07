# swarm-strike-allocation

面向多群分散入侵目标的无人机集群打击任务分配——统一仿真与算法比较平台。

大创项目（2025.11–2026.12）。作战想定：半球形战场（R=3~5 km），敌方多类型无人机分波次（0、T/3、2T/3 s）直奔基地，我方以同质自杀式无人机"以集群反集群"，多打一拦截。在**统一模型 + 算法接口层**框架下公平比较 8 个候选任务分配算法。

## 框架

```
场景生成器(50场景·固定种子) ──► 感知层(增量DBSCAN分组 → 熵权TOPSIS威胁评估)
                                      │
                                      ▼
        ┌────── 统一模型 model ──────┐
        │ 毁伤公式·配额·目标函数·约束 │
        └─────────────┬──────────────┘
                      ▼ solve(state, model) → X
        算法接口层（greedy / ga_iap / cbba_ye / … 每人一个，即插即用）
                      │
                      ▼
        统一评估（a收益·b威胁消除率·c稳定性·d效费比·e耗时）→ 50场景加权总分
```

**铁律**：算法只读 `state`、只返回分配矩阵 `X`；毁伤、配额、约束、指标一律调用 `model` / `evaluation`，不在算法内部另写。

## 环境配置

```bash
pip install -e .          # 安装依赖（pyproject.toml 锁定）
pip install -e .[dev]     # 含 pytest 等开发依赖
```

要求 Python ≥ 3.11。

## 快速开始

```bash
python experiments/exp1_main.py     # 主实验：全部算法 × 50 场景 → results/
pytest tests/                       # 改代码前必跑单元测试
```

## 如何添加新算法

1. 复制 `src/swarm_alloc/algorithms/greedy.py` 为 `algorithms/<你的算法>.py`；
2. 继承 `AlgoBase` 并加 `@register` 装饰器，实现 `solve(state, model) -> X`（X 为 0-1 分配矩阵）；
3. 无需手动注册：`algorithms/__init__.py` 会自动发现并导入本目录所有算法模块；
4. 实验脚本即可按名字调用（`exp0_smoke.py --algo <名>` / `exp1_main.py --algos <名>`），无需改动框架。

## 参数修改入口

所有参数集中在 `config/`，**代码中禁止硬编码**：

| 文件 | 内容 |
|---|---|
| `scenario.yaml` | 战场半径、波次、每波群数、生成种子表 |
| `enemy_types.yaml` | 敌机类型表（对基地/对拦截机/对信息：价值、机动、p_j、杀伤参数） |
| `friendly.yaml` | 我方数量、速度、单机价值 |
| `eval.yaml` | 歼灭阈值 0.9、指标权重 A~D、耗时惩罚系数 |

## 分工与里程碑

| 里程碑 | 内容 | 产出 | 负责人 |
|---|---|---|---|
| M1 | config + 场景生成器 | `data/scenes.pkl`（全组共用） | 待定 |
| M2 | 感知层 + 统一模型 + greedy baseline | 可跑通闭环 | 待定 |
| M3 | 仿真主循环 + 评估模块 | 单算法×50场景指标表 | 待定 |
| M4 | 算法分工（先进 ga_iap，见 docs/） | 各算法工作文档 | 每人1~2个 |
| M5 | exp1~exp3 + 统计分析 | 技术报告图表 | 待定 |

## 目录说明

详见 `../工程实现方案大纲.md`。候选算法清单与场景参数以 `docs/方案04-v3.docx` 为准，模型数学定义见 `docs/统一模型文档v1.md`。
