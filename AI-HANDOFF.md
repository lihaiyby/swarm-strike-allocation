# AI 接手文档 · swarm-strike-allocation

> 写给接手本工程的 AI（或新会话）：读完本文即可开始复现下一篇文献算法。
> 本文只讲"怎么接手干活"，场景与模型的权威定义以 `方案04-v3.docx` 和 `docs/统一模型文档v1.md` 为准。

---

## ⚠️ 先读这里（2026-10-23 起生效）

本工程已建立**四件套协作体系**。任何新会话请按顺序读：

| 顺 | 文件 | 作用 |
|---|---|---|
| 0 | [`AGENTS.md`](AGENTS.md) | **强制规约**：开工读四件套、收工更四件套、关键改动必须推 GitHub |
| 1 | [`docs/擂台改造-任务清单.md`](docs/擂台改造-任务清单.md) | 现在该做哪一条（含 5 个待拍板决策项 D1–D5） |
| 2 | [`docs/擂台改造-进度记录.md`](docs/擂台改造-进度记录.md) | 上一轮做到哪、有哪些结论被修订过 |
| 3 | [`docs/变量及参数及符号大全.md`](docs/变量及参数及符号大全.md) | 每个符号/参数/字段的含义与使用环节；§10 有"定义了但未生效"清单 |
| 4 | [`docs/AI工作日志.md`](docs/AI工作日志.md) | 最近的改动、commit 与 GitHub 推送状态 |

> **背景变更提示**：经实测，现行模型在 3n 编制下 `greedy` 已贴住 (a,b) 的理论天花板，算法之间缺乏区分度；且指标 `b` 的口径（动态重聚类 vs 真值团）会显著改变结论。**在阅读本文 §6 的基线成绩时请一并阅读上述文档**，其中 `docs/擂台改造-进度记录.md` 的"结论修订表"记录了哪些旧结论已被推翻。
>
> 本文 §7 的算法复现清单仍然有效，但建议在任务清单 **P0（口径修复）** 完成后再开跑，否则各算法结果不可比。

---

## 0. 一句话项目背景

大创项目：半球形战场（R=3~5 km），敌方多类型无人机分 3 个波次（0、T/3、2T/3 s，每波 6 群，共 n=100~150 架）直奔基地，我方 3n 架同质自杀式无人机"以集群反集群"多打一拦截。本工程是**统一仿真与算法比较平台**：场景、感知、模型、评估全部固定，算法即插即用，公平比较。已完成复现01（IAP-NSGA-III，注册名 `ga_iap`），接下来按 §7 清单继续复现其他文献算法。

## 1. 环境与运行（先验证能跑，再动手）

- Python：`D:\python\python.exe`（3.13，依赖已装好）。**不要用**受管运行时或 venv 里的 python（损坏，报 error 103）。
- 工程未 `pip install -e`，运行时设 `PYTHONPATH=src`。
- Git Bash 对中文路径支持差（`cd /e/无人机集群` 会失败），涉及中文路径的操作一律用 `D:\python\python.exe` 脚本完成。

```bash
cd "E:\A无人机集群\打击目标分配\swarm-strike-allocation"

# 1) 单元测试（动手前必跑，当前 15 项全过）
PYTHONPATH=src "D:\python\python.exe" -m pytest tests/ -q

# 2) 单场景冒烟（新算法调试用）
PYTHONPATH=src "D:\python\python.exe" experiments/exp0_smoke.py --algo <注册名>

# 3) 主实验：算法 × 50 场景 → results/exp1/
PYTHONPATH=src "D:\python\python.exe" experiments/exp1_main.py --algos greedy ga_iap <新算法> --scenes 50
```

## 2. 目录地图与改动边界

```
swarm-strike-allocation/
├── config/               # 全部参数（scenario/enemy_types/friendly/eval.yaml），代码禁止硬编码
├── data/scenes.pkl       # 50 个场景（种子 1000~1049），全组共用，勿重新生成
├── docs/
│   ├── 统一模型文档v1.md          # 模型数学定义（毁伤/配额/目标/约束/指标）
│   └── algorithms/复现01-IAP-NSGA-III.docx   # 算法工作文档模板（八节结构）
├── src/swarm_alloc/
│   ├── scenario/         # 场景生成 + 可视化      【只读】
│   ├── perception/       # 增量DBSCAN分组 + 熵权TOPSIS威胁评估  【只读】
│   ├── model/core.py     # 统一模型 Model 类      【只读】
│   ├── simulation/       # State / 战损结算 / 主循环 simulate  【只读】
│   ├── evaluation/       # 指标 a~e               【只读】
│   └── algorithms/       # ★ 你唯一需要新增代码的地方
│       ├── base.py       # AlgoBase + @register 注册表
│       ├── greedy.py     # 边际收益贪心 baseline（模仿对象）
│       └── ga_iap.py     # 复现01：IAP-NSGA-III 适配版
├── experiments/          # exp0 冒烟 / exp1 主实验  【可加实验脚本，勿改 exp1 的统计口径】
├── tests/                # 单测；改框架代码前必须全绿
└── results/exp1/         # 实验产出（csv + 图）
```

**边界铁律**：新算法 = 在 `algorithms/` 新增一个文件（类上加 `@register` 装饰器），`algorithms/__init__.py` 会自动发现注册，**无需改任何实验脚本的 import**。**不允许**改 model / evaluation / scenario / perception / simulation 的任何逻辑。发现框架 bug 时先写失败测试复现，再修复，并在工作文档中记录。

## 3. 算法接口契约（最重要的一节）

```python
from ..model import Model
from .base import AlgoBase, register

@register
class MyAlgo(AlgoBase):
    name = "my_algo"          # 注册名：小写下划线，exp 脚本按此调用

    def solve(self, state, model: Model) -> np.ndarray:
        # 只读 state，只返回 X；禁止在内部另写毁伤/配额/约束/指标公式
        ...
        return X              # (len(state.pool_ids) × len(state.enemy_ids)) 的 0-1 int 矩阵
```

**State 提供**（`simulation/state.py`）：

| 成员 | 含义 |
|---|---|
| `state.t` | 当前时刻（s） |
| `state.enemies` / `state.enemy_ids` | 在空敌机（id → EnemyUnit，含 pos/value/p_kill/类型） |
| `state.groups` | 感知层分好的敌群（Group，含 member_ids、threat、质心） |
| `state.pool` / `state.pool_ids` | 可用我方无人机（id → FriendlyUnit，含 pos） |
| `state.frozen_ids` | 已下达指令不可改派的我机（C5 约束） |
| `state.enemy_matrix()` | → (values, p_kill, positions)，按 enemy_ids 顺序 |
| `state.pool_matrix()` | → 我机位置矩阵，按 pool_ids 顺序 |

**Model 提供**（`model/core.py`，必须调用而非重写）：

| 方法 | 含义 |
|---|---|
| `Model.damage(p, k)` | 累积毁伤 P(k)=1−(1−p)^k |
| `Model.quota(p, threshold)` | 配额 m_j=⌈ln(1−阈值)/ln(1−p_j)⌉（p=0.4/0.7/0.9 → 5/2/1 架） |
| `model.objective(X, values, p_kill)` | 目标函数 = 指标 a，Σ v_j·P_j |
| `model.check(X, state, quotas, group_of_enemy, group_of_friendly)` | 约束 C1~C5 检查，返回违规列表（空=可行） |
| `Model.select_groups(groups, demand, pool_size)` | 可行性恢复：资源不足时按威胁度纳入、放弃低威胁群 |
| `model.kill_threshold` / `model.time_limit_s` | 歼灭阈值 0.9 / 单轮耗时上限 1.0 s |

**X 的语义**：`X[i,j]=1` 表示 `pool_ids[i]` 打击 `enemy_ids[j]`。约束要点：C1 单架单目标；C2 每目标要么 0 架要么 ≥配额；C3 群对群（编组内我机只能打本群目标）；C5 冻结机不可分配。配额同时也是**搜索上限**（防过杀耗尽资源池，复现01 确立的原则）。

## 4. 复现流程（五步法）

1. **读文献**：PDF 在 `E:\A无人机集群\文献-老师给的\`。提取文本：
   `"D:\python\python.exe" -c "import subprocess,sys; r=subprocess.run(['pdftotext', r'<pdf路径>', '-'], capture_output=True); sys.stdout.buffer.write(r.stdout)" | more`
   （或先用 `E:\A无人机集群\文献-老师给的\_extracted\` 里已提取的 txt）。先读 `算法总结与难度分类.md` 中对应条目定位重点章节。
2. **写适配改动点**：原文模型 → 本场景的映射（单机/多机、异构→同质、目标函数替换为 model.objective、哪些机制保留/略去及理由）。先写成几行笔记，和用户确认后再写代码。
3. **实现 solve**：复制 `greedy.py` 为新文件，改类名/注册名（`@register` 自动被发现，无需动实验脚本），实现算法。群级配对可复用 `Model.select_groups`；需要编组划分参考 greedy 的"离群质心最近"取机方式。
4. **单场景验证**：跑 `exp0_smoke.py --algo <注册名>`，确认 `model.check` 无违规、单轮耗时 < 1 s（指标 e 的硬上限）；跑通后 `pytest tests/ -q` 全绿。
5. **50 场景对比**：exp1 加入新算法，与 greedy / ga_iap 同场对比，产出 csv 与图到 `results/exp1/`。

## 5. 已踩过的坑（别再踩）

- **评估口径是期望值**：不掷骰子，毁伤一律用 P(k)=1−(1−p)^k 的期望（方案04-v3"口径 A"）。
- **全歼后提前 break 的 bug 已修**：simulate 主循环不要在敌机清空时直接跳出，后续波次还要进场。
- **TOPSIS 单群除零**：熵权法 log(1) 已做保护，改动 perception 时注意。
- **噪点数量要随机**：场景生成中散点（占 5%）用二项抽样，不要 round() 恒定值。
- **matplotlib 出图用英文标签**，避免 CJK 字体缺失方块。
- **b 指标与目标函数的张力**：GA 类算法若按纯收益优化会放弃低价值侦察机群，导致 b（群级威胁消除率）低于 greedy——这是 doctrine 层面的取舍，不是 bug。
  > ⚠️ **口径已于 2026-10-25 变更（P0.1/P0.6，决策 D2=②）**：b 现按 **真值团（`Scene.ground_truth_groups()`）+ 全员阈值 + 价值加权**、**战局结束一次性**计算，唯一实现位于 `evaluation/b_metric.py`。旧的 `threat_total`/`seen_groups` 动态聚类口径已删除——它会让分母随算法行为漂移（噪点每回合领新 ID），同场景 greedy 21 个群 ID vs ga_iap 53 个（真实团数 21）。
  > 变更后 ga_iap 的 b 由 **0.405 变为 0.889**（中位 0.938，仅 12/50 场景 < 0.8），**"系统性放弃侦察机"的旧结论只对了一半**，详见 `docs/擂台改造-进度记录.md` 2026-10-25 条目与任务 R3。
- **docx 产出**：本机无 LibreOffice，无法渲染检查 docx；构建用 kimi-word skill 的 `docx.py build`，已有构建脚本范例 `E:\A无人机集群\打击目标分配\_build复现01.js`。

## 6. 基线成绩（50 场景，mean±std，用于对比）

> b 列已于 2026-10-25 按新口径（真值团 + 价值加权）重跑；a、d 不变。

| 算法 | a 打击总收益 | b 威胁消除率 | d 效费比 | 单场景耗时 |
|---|---|---|---|---|
| greedy | 589.7 ± 63.2 | **0.9983 ± 0.0071** | 1.657 ± 0.126 | 6 ms |
| ga_iap（复现01） | 584.8 ± 63.0 | **0.8887 ± 0.1487** | 1.714 ± 0.145 | 3.9 s（固定 200 代，单轮 ~0.75 s） |

配对效应量：a 的 Cohen's d = −0.65（greedy 胜，5/50 场景 ga 更好）、b = **−0.75**（greedy 胜，0/50）、d = **+0.71**（ga 胜，33/50）。
→ **greedy 赢 a、b，ga_iap 赢 d**，是健康的三方权衡，而非单方碾压。

新算法的目标：a、d 不低于 greedy 太多，b 显著高于 ga_iap（即不放弃低价值群），单轮耗时 < 1 s。

## 7. 待复现算法清单（按优先级，详见 `文献-老师给的\算法总结与难度分类.md`）

| 优先级 | 文献 | 算法 | 建议注册名 | 状态 |
|---|---|---|---|---|
| — | 刘富樯 | IAP-NSGA-III | `ga_iap` | ✅ 复现01 已完成 |
| P1 | 叶赞 | 改进 CBBA（任务包双向构建+类别奖励） | `cbba_ye` | ⬜ 下一个 |
| P1 | 常绪成 | 改进 DBSCAN + 联盟形成博弈（与项目路线同构） | `dbscan_coalition` | ⬜ |
| P1 | 商鹏 | TR-CBBA 威胁重置捆绑（对应动态重分配） | `tr_cbba` | ⬜ |
| P2 | 毕义军 | 联盟博弈 + 贪心预分配 + 改进模拟退火 | `pc_isa` | ⬜ |
| P2 | 张淳 | Gale-Shapley 稳定匹配 + Shapley 值 | `gs_shapley` | ⬜ |
| P2 | 孙鹏耀 | 改进双种群遗传算法 | `ga_dual` | ⬜ |
| P2 | 文凤鹏 | SSDR-CBBA（次模+影子价格） | `ssdr_cbba` | ⬜ |
| P3 | 曹筱可 | 享乐博弈分簇 + 层次化分配 | `hedonic` | ⬜ |

强化学习类（王旭 DQN、李松 DQN、冯月姣分层 PPO 等）暂不纳入本平台，如需做另开实验分支。

## 8. 完成一个算法后要交付什么

1. `src/swarm_alloc/algorithms/<name>.py`（含注册）；
2. `results/exp1/` 下含该算法的对比 csv 与图；
3. **算法工作文档** `docs/algorithms/复现0N-<算法名>.docx`，沿用复现01 的八节结构：
   ① 文献信息 ② 原文算法概述 ③ 与本项目的场景差异 ④ 适配改动点（含略去机制的理由） ⑤ 实现接口说明 ⑥ 实验结果（50 场景指标表+图） ⑦ 核心发现 ⑧ 结论。
4. 更新本文件 §7 的状态列。

## 9. 速查

| 问题 | 去哪看 |
|---|---|
| 场景/模型为什么这么定 | `方案04-v3.docx`（工程根目录）、`docs/统一模型文档v1.md` |
| 接口怎么写 | `src/swarm_alloc/algorithms/greedy.py`（60 行，先读它） |
| 参数含义 | `config/*.yaml`（均有注释） |
| 复现01 怎么做的 | `docs/algorithms/复现01-IAP-NSGA-III.docx` |
| 哪篇文献值得做 | `E:\A无人机集群\文献-老师给的\算法总结与难度分类.md` |
| 仿真主循环逻辑 | `src/swarm_alloc/simulation/simulate.py` |
