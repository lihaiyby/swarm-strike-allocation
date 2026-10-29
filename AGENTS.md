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
| `git push` 报 `Connection was reset` / `Failed to connect ... port 443` | **不是网络不通**：本机系统代理 `127.0.0.1:7897` 开着，Edge 走它所以能上 GitHub；但 **git 不读 Windows Internet 设置**，`HTTP_PROXY`/`HTTPS_PROXY`/`git http.proxy` 皆空 → git 直连，TCP 能通、TLS 被重置 | 显式指定代理，见 §6.3 |
| `git push` 报 `couldn't create signal pipe, Win32 error 5` 或 `could not read Username` | 沙箱禁止创建命名管道，git 凭据助手（Windows 凭据管理器）无法运行 | **以完全权限推送** |
| `git commit` 报 `Unable to create .git/index.lock: Permission denied` | DSH 沙箱的受限令牌写不进 `.git`（ACL 已补当前用户完全控制并验证，仍被拒） | **以完全权限执行 `git add/commit`** |
| `git clone/push` 报 `SEC_E_NO_CREDENTIALS` | 本机 `schannel` TLS 后端不可用 | 一律加 `-c http.sslBackend=openssl` |
| `pytest` 报 `could not create cache path ... Permission denied` | 残留 `pytest-cache-files-*` 目录 | 已清理；如再出现，删除该目录即可 |
| 命令突然变得极慢：`import numpy` 要 30~40 s、正常只需 0.6 s | **Windows 安全中心（`SecurityHealthHost.exe`）多实例卡死**（正常 1 个，异常时 6 个，各累计数百秒 CPU），把磁盘 I/O 堵死。触发条件：短时间生成大量新文件（本项目每轮会写 pkl + 数十张 PNG） | ① `Stop-Process -Name SecurityHealthHost -Force`（会自动重启）；② 无效则重启；③ 长期方案见 §6.2 排除项 |

### 6.2 本机性能设置（2026-10-29 已配置）

为避免上述安全中心问题复现，已为开发目录添加 Defender 排除项（**实时保护保持开启**）：

| 排除路径 | 说明 |
|---|---|
| `D:\deepseek-workspace` | 本项目与全部工作区 |
| `D:\python` | Python 安装与 site-packages |
| `%LOCALAPPDATA%\pip\Cache` | pip 缓存 |

- 作用：**`import numpy+pandas+matplotlib` 从 4.39 s → 0.62 s（约 7×）**，`pytest` 全量 7.25 s。
- 未添加扩展名/进程级排除（`.pkl`、`python.exe`）——那会显著降低保护，本次刻意不做。
- 复现命令（**需管理员**；本机 UAC 为 `ConsentPromptBehaviorAdmin=0`，管理员提权不弹窗）：
  ```powershell
  Start-Process powershell -Verb RunAs -Wait -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','D:\deepseek-workspace\setup_defender_exclusions.ps1'
  ```
- 该脚本与运行日志放在仓库外的 `D:\deepseek-workspace\`（`setup_defender_exclusions.ps1`、`_defender_setup_log.txt`）。
- ⚠️ 脚本必须是**纯 ASCII**：Windows PowerShell 5.1 会把无 BOM 的 UTF-8 当 ANSI 读，含中文会导致解析失败（exit 1、无日志）。

### 6.3 推送命令（复制即用）

**先判断代理是否开启**：

```powershell
(Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings').ProxyEnable
```

| `ProxyEnable` | 用哪条 |
|---|---|
| **1（开启）** | 必须显式给 git 指定代理（git 不读 Windows 的 Internet 设置） |
| **0（关闭）** | 直连即可；若直连被重置，再按开启处理 |

```bash
# 代理开启时
git -c http.proxy=http://127.0.0.1:7897 \
    -c https.proxy=http://127.0.0.1:7897 \
    -c http.sslBackend=openssl push origin main

# 代理关闭时（2026-10-29 实测可用）
git -c http.sslBackend=openssl push origin main
```

> `http.sslBackend=openssl` **两种情况都必须带**（`schannel` 会报 `SEC_E_NO_CREDENTIALS`）。
> 代理端口若变更，改 `7897` 即可；查当前端口：
> `Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings' | Select ProxyServer`

---

## 7. 当前优先级（2026-10-29 更新）

1. **R3 修订复现01 文档** —— 欠账最久。该文档 §7 的核心结论（b 由 0.996 跌至 0.397）已被证伪：新口径下 greedy **0.998** vs ga_iap **0.889**。建议先出 markdown 修订稿再回写 docx。
2. **D3 定评估权重 A~E** —— `eval.yaml` 五个权重仍是 `null`。a 与 b 各有冠军、档位还会让综合冠军翻转，不定权重就出不了唯一排名。
3. **P3.5 消融 / 灵敏度** —— 配比、威胁度、杀伤参数各一组对照。
4. 未做：P0.4（X 守卫）、P0.5 的 ③④、D4（C3）、D5（时间耦合）。

> 详细依赖图见任务清单 §9；当前进度见进度记录顶部快照。
