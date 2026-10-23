# AGENTS.md · AI 协作规约

> 本文件是**给 AI 的强制规约**，优先级高于其他文档中的操作建议。
> 适用对象：任何接手本仓库的 AI 会话（Claude / GPT / DeepSeek / Copilot 等）。
> 建立日期：2026-10-23

---

## 0. 一句话

**开工先读四件套，收工必更新四件套，并推到 GitHub。**

---

## 1. 开工前必读（按顺序，勿跳）

| 顺 | 文件 | 读它干什么 |
|---|---|---|
| 1 | [`AGENTS.md`](AGENTS.md) | 本文件，先确认规矩 |
| 2 | [`docs/擂台改造-任务清单.md`](docs/擂台改造-任务清单.md) | 现在该做哪一条、依赖是什么、验收标准是什么 |
| 3 | [`docs/擂台改造-进度记录.md`](docs/擂台改造-进度记录.md) | 上一轮做到哪、结论是什么、有没有被推翻的判断 |
| 4 | [`docs/变量及参数及符号大全.md`](docs/变量及参数及符号大全.md) | 每个名字指什么、在哪个环节生效、哪些是闲置的 |
| 5 | [`docs/AI工作日志.md`](docs/AI工作日志.md) | 最近几次改动与推送状态 |
| 6 | [`AI-HANDOFF.md`](AI-HANDOFF.md) | 工程结构、接口契约、已踩过的坑 |

> 只读 1~5 就能恢复上下文；6 是项目原始接手文档，含更细的踩坑记录。

---

## 2. 动手时的边界

- **唯一自由新增代码的地方**是 `src/swarm_alloc/algorithms/`（新算法一个文件 + `@register`，自动发现）。
- **改动框架代码**（`model/` `perception/` `simulation/` `evaluation/` `scenario/` `config/`）**必须先**在任务清单里登记一条任务，说明动机与验收标准，再动手。
- **参数只改 `config/*.yaml`**，代码中禁止硬编码。
- 改代码前先 `pytest tests/ -q` 建立基线；改完必须仍然全绿。

---

## 3. 收工前必做：四件套更新

**这是硬性要求。任何一次会话结束前，逐项确认：**

| # | 文件 | 什么时候必须更新 | 更新什么 |
|---|---|---|---|
| **1** | [`docs/擂台改造-任务清单.md`](docs/擂台改造-任务清单.md) | **每次** | 任务状态（⬜🟨✅⛔⏸）；新增任务按 `D/P0/P1/P2/P3/A/R/M` 编号；完成的填验收结果 |
| **2** | [`docs/擂台改造-进度记录.md`](docs/擂台改造-进度记录.md) | **每次** | 顶部"当前快照"刷新；追加一条倒序日志（目标/做法/发现/证据/影响） |
| **3** | [`docs/变量及参数及符号大全.md`](docs/变量及参数及符号大全.md) | **只要有名字变动** | 新增/重命名/删除的符号、参数、字段、配置项；§10"未使用清单"同步增删 |
| **4** | [`docs/AI工作日志.md`](docs/AI工作日志.md) | **每次** | 追加一条 7 字段日志；更新"待推送队列"与"GitHub 同步状态总览" |

### 自检清单（往下抄，逐条打勾）

```
[ ] 任务清单：本次涉及的任务状态已改，验收结果已写
[ ] 进度记录：当前快照已刷新，新日志条目已追加
[ ] 变量大全：有无新增/改名/删除的变量？有就改，没有就在日志里写明"无符号变动"
[ ] AI工作日志：已追加条目，含 commit hash 与推送状态
[ ] pytest 全绿
[ ] 已 git commit
[ ] 已 git push（或登记到待推送队列并说明原因）
```

---

## 4. Git 提交与推送规范

### 4.1 必须推送

**关键修改和重要工作必须推到 GitHub。** 不允许只在本地留改动就结束会话。

```bash
git add -A
git commit -m "<type>: <一句话概括>"
git -c http.sslBackend=openssl push origin main
```

> ⚠️ 本机 `http.sslBackend=schannel` 会报 `SEC_E_NO_CREDENTIALS`，**推送时务必显式指定 openssl 后端**。

### 4.2 commit message 格式

`<type>: <概括>`，type 取：

| type | 用于 |
|---|---|
| `feat` | 新增功能（新算法、新模块） |
| `fix` | 修 bug（含口径修复） |
| `docs` | 纯文档 |
| `refactor` | 不改行为的重构 |
| `test` | 新增/修改测试 |
| `exp` | 实验脚本与结果 |
| `chore` | 杂项（依赖、配置、清理） |

例：`fix: 指标 b 改用真值团，消除噪点重编号导致的分母膨胀`

### 4.3 推送失败怎么办

1. **不要伪造成功**。把真实原因写进 `docs/AI工作日志.md` 的"待推送队列"；
2. 记录：commit hash、message、失败原因（超时/认证/冲突）；
3. 在网络恢复后，**下一个会话有义务先补推**，再开始自己的任务；
4. 补推成功后，把队列条目标为 ✅ 并写明推送时间。

### 4.4 推送前检查

```bash
git status --short          # 确认没有意外文件（_probe/ 等临时目录不应入库）
git log --oneline -3        # 确认提交信息正确
```

> `.gitignore` 已忽略 `data/*.pkl` 之外的部分产物；`_probe/` 是分析用临时目录，**不入库**。

---

## 5. 日志条目模板（复制到 `docs/AI工作日志.md` 顶部）

```markdown
### YYYY-MM-DD HH:MM · <一句话标题>

**任务**：<对应任务清单 ID + 要解决什么>

**改动**
| 文件 | 类型 | 说明 |
|---|---|---|
| `path/to/file` | 新增/修改/删除 | ... |

**验证**：<命令 + 结果，含关键数字>

**结论**：<事实性结论，不是"已完成">

**提交**：`<hash>`
**推送**：✅/⏳ <状态与原因>
```

---

## 6. 快速自检命令

```bash
cd swarm-strike-allocation
PYTHONPATH=src "D:\python\python.exe" -m pytest tests/ -q
git status --short
git log --oneline -3
```

### 6.1 本机已知环境障碍（2026-10-23 实测）

| 症状 | 原因 | 处理 |
|---|---|---|
| `git commit` 报 `Unable to create .git/index.lock: Permission denied` | DSH 沙箱的受限令牌写不进 `.git`（ACL 已补当前用户完全控制并验证，仍被拒） | **以完全权限（danger-full-access）执行 `git add/commit`** |
| `git clone/push` 报 `SEC_E_NO_CREDENTIALS` | 本机 `schannel` TLS 后端不可用 | 一律加 `-c http.sslBackend=openssl` |
| `git push` 报 `Failed to connect to github.com port 443` / `Connection was reset` | 网络层不可达（非本地配置问题，重试无效） | 登记到 `docs/AI工作日志.md` 待推送队列，**下个会话先补推** |
| `pytest` 报 `could not create cache path ... Permission denied` | 残留 `pytest-cache-files-*` 目录 | 已清理；如再出现，删除该目录即可 |

---

## 7. 当前优先级（2026-10-23）

1. **D2 待拍板**：指标 b 用哪种群定义（推荐真值团）→ 解锁 P0.1/P0.6/R3；
2. **P0.3 可立即做**：ga_iap 去墙钟截断（零依赖）；
3. 之后按 `docs/擂台改造-任务清单.md` §9 的顺序推进。

> 详细依赖图见任务清单 §9。
